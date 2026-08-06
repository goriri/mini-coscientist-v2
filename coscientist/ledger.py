"""Transactional SQLite persistence for research sessions and audit events."""

from __future__ import annotations

import hmac
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import AuditEvent, Session, utc_now


class ConcurrentSessionUpdate(RuntimeError):
    """Raised when a stale process attempts to overwrite a newer session."""


class ResearchLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    delete_token_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS audit_session_sequence
                    ON audit_events(session_id, sequence);
                CREATE TABLE IF NOT EXISTS workflow_operations (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)")
            }
            if "delete_token_hash" not in columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN delete_token_hash TEXT"
                )

    def save(
        self,
        session: Session,
        *,
        expected_version: int | None = None,
        event: AuditEvent | None = None,
    ) -> None:
        prior_version = (
            session.version if expected_version is None else expected_version
        )
        next_version = prior_version + 1
        session.updated_at = utc_now()
        session.version = next_version
        payload = session.canonical_json()
        try:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT version FROM sessions WHERE id = ?", (session.id,)
                ).fetchone()
                if existing is None:
                    if prior_version != 0:
                        raise ConcurrentSessionUpdate(
                            f"Session {session.id} does not exist at version {prior_version}."
                        )
                    connection.execute(
                        "INSERT INTO sessions(id, version, payload, updated_at) VALUES (?, ?, ?, ?)",
                        (session.id, next_version, payload, session.updated_at),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE sessions SET version = ?, payload = ?, updated_at = ?
                        WHERE id = ? AND version = ?
                        """,
                        (
                            next_version,
                            payload,
                            session.updated_at,
                            session.id,
                            prior_version,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ConcurrentSessionUpdate(
                            f"Session {session.id} changed after version {prior_version}."
                        )
                if event is not None:
                    self._insert_event(connection, session.id, event)
        except Exception:
            session.version = prior_version
            raise

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, session_id: str, event: AuditEvent
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO audit_events(
                session_id, event_id, event_type, actor, stage, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event.id,
                event.event_type,
                event.actor,
                event.stage,
                json.dumps(event.payload, sort_keys=True),
                event.created_at,
            ),
        )

    def append_event(self, session_id: str, event: AuditEvent) -> None:
        with self._connect() as connection:
            self._insert_event(connection, session_id, event)

    def load(self, session_id: str) -> Session:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return Session.from_dict(json.loads(row[0]))

    def events(self, session_id: str) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, event_type, actor, stage, payload, created_at
                FROM audit_events WHERE session_id = ? ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return [
            AuditEvent(
                id=row[0],
                event_type=row[1],
                actor=row[2],
                stage=row[3],
                payload=json.loads(row[4]),
                created_at=row[5],
            )
            for row in rows
        ]

    def set_delete_token_hash(self, session_id: str, token_hash: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET delete_token_hash = ? WHERE id = ?",
                (token_hash, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown session: {session_id}")

    def delete_session(self, session_id: str, token_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT delete_token_hash FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None or not row[0] or not hmac.compare_digest(row[0], token_hash):
                return False
            connection.execute(
                "DELETE FROM workflow_operations WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM audit_events WHERE session_id = ?", (session_id,)
            )
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

    def set_operation(
        self, session_id: str, status: str, detail: str, kind: str
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_operations(
                    session_id, status, detail, kind, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    detail = excluded.detail,
                    kind = excluded.kind,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (session_id, status, detail, kind, now),
            )

    def operation(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, detail, kind, lease_owner, lease_expires_at,
                       attempt, updated_at
                FROM workflow_operations WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return {"status": "idle", "detail": "", "kind": "generation"}
        return {
            "status": row[0],
            "detail": row[1],
            "kind": row[2],
            "lease_owner": row[3],
            "lease_expires_at": row[4],
            "attempt": row[5],
            "updated_at": row[6],
        }

    def claim_operation(
        self,
        session_id: str,
        owner: str,
        *,
        detail: str,
        lease_seconds: int = 300,
    ) -> bool:
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_operations
                SET status = 'running', detail = ?, lease_owner = ?,
                    lease_expires_at = ?,
                    attempt = attempt + 1, updated_at = ?
                WHERE session_id = ?
                  AND (
                    status = 'queued'
                    OR (
                        status = 'running'
                        AND (lease_expires_at IS NULL OR lease_expires_at < ?)
                    )
                  )
                """,
                (
                    detail,
                    owner,
                    expires,
                    now.isoformat(),
                    session_id,
                    now.isoformat(),
                ),
            )
            return cursor.rowcount == 1

    def renew_operation(
        self,
        session_id: str,
        owner: str,
        *,
        detail: str | None = None,
        lease_seconds: int = 300,
    ) -> bool:
        """Extend this worker's lease, and optionally say what it is waiting on.

        A worker that works or waits longer than its lease -- an evidence stage
        polls every Deep Research pass to completion inside one call -- would
        otherwise be declared dead by ``requeue_expired_operation`` and have a
        second worker started beside it. Returns False if the lease has already
        been taken away, which is the signal to stop rather than to carry on
        writing to a session somebody else now owns.

        ``detail`` left unset keeps the message already on the operation, so a
        heartbeat can hold the lease without overwriting what the worker last
        told the researcher it was doing.
        """
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_operations
                SET detail = COALESCE(?, detail), lease_expires_at = ?,
                    updated_at = ?
                WHERE session_id = ? AND status = 'running' AND lease_owner = ?
                """,
                (detail, expires, now.isoformat(), session_id, owner),
            )
            return cursor.rowcount == 1

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone() == (1,)

    def requeue_expired_operation(self, session_id: str) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_operations
                SET status = 'queued', detail = 'Recovering interrupted work.',
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE session_id = ? AND status = 'running'
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now, session_id, now),
            )
            return cursor.rowcount == 1


class PostgresResearchLedger:
    """PostgreSQL implementation used by horizontally scaled Cloud Run."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._initialize()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DO $migration$
                BEGIN
                    IF to_regclass('public.research_sessions') IS NULL
                       AND EXISTS (
                           SELECT 1
                           FROM information_schema.columns
                           WHERE table_schema = 'public'
                             AND table_name = 'sessions'
                             AND column_name = 'payload'
                       )
                    THEN
                        ALTER TABLE sessions RENAME TO research_sessions;
                    END IF;
                    IF to_regclass('public.research_audit_events') IS NULL
                       AND to_regclass('public.audit_events') IS NOT NULL
                    THEN
                        ALTER TABLE audit_events RENAME TO research_audit_events;
                    END IF;
                    IF to_regclass('public.research_workflow_operations') IS NULL
                       AND to_regclass('public.workflow_operations') IS NOT NULL
                    THEN
                        ALTER TABLE workflow_operations
                            RENAME TO research_workflow_operations;
                    END IF;
                END
                $migration$
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    delete_token_hash TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_audit_events (
                    sequence BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES research_sessions(id),
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS audit_session_sequence
                ON research_audit_events(session_id, sequence)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_workflow_operations (
                    session_id TEXT PRIMARY KEY REFERENCES research_sessions(id),
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )

    def save(
        self,
        session: Session,
        *,
        expected_version: int | None = None,
        event: AuditEvent | None = None,
    ) -> None:
        prior_version = (
            session.version if expected_version is None else expected_version
        )
        next_version = prior_version + 1
        session.updated_at = utc_now()
        session.version = next_version
        payload = session.canonical_json()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT version FROM research_sessions WHERE id = %s FOR UPDATE",
                    (session.id,),
                )
                existing = cursor.fetchone()
                if existing is None:
                    if prior_version != 0:
                        raise ConcurrentSessionUpdate(
                            f"Session {session.id} does not exist at version {prior_version}."
                        )
                    cursor.execute(
                        """
                        INSERT INTO research_sessions(
                            id, version, payload, updated_at
                        )
                        VALUES (%s, %s, %s::jsonb, %s)
                        """,
                        (session.id, next_version, payload, session.updated_at),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE research_sessions
                        SET version = %s, payload = %s::jsonb,
                            updated_at = %s
                        WHERE id = %s AND version = %s
                        """,
                        (
                            next_version,
                            payload,
                            session.updated_at,
                            session.id,
                            prior_version,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ConcurrentSessionUpdate(
                            f"Session {session.id} changed after version {prior_version}."
                        )
                if event is not None:
                    self._insert_event(cursor, session.id, event)
        except Exception:
            session.version = prior_version
            raise

    @staticmethod
    def _insert_event(cursor, session_id: str, event: AuditEvent) -> None:
        cursor.execute(
            """
            INSERT INTO research_audit_events(
                session_id, event_id, event_type, actor, stage, payload, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (
                session_id,
                event.id,
                event.event_type,
                event.actor,
                event.stage,
                json.dumps(event.payload, sort_keys=True),
                event.created_at,
            ),
        )

    def append_event(self, session_id: str, event: AuditEvent) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            self._insert_event(cursor, session_id, event)

    def load(self, session_id: str) -> Session:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM research_sessions WHERE id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return Session.from_dict(payload)

    def events(self, session_id: str) -> list[AuditEvent]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_id, event_type, actor, stage, payload, created_at
                FROM research_audit_events
                WHERE session_id = %s ORDER BY sequence
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
        return [
            AuditEvent(
                id=row[0],
                event_type=row[1],
                actor=row[2],
                stage=row[3],
                payload=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                created_at=row[5].isoformat(),
            )
            for row in rows
        ]

    def set_delete_token_hash(self, session_id: str, token_hash: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_sessions
                SET delete_token_hash = %s WHERE id = %s
                """,
                (token_hash, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown session: {session_id}")

    def delete_session(self, session_id: str, token_hash: str) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT delete_token_hash FROM research_sessions
                WHERE id = %s FOR UPDATE
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None or not row[0] or not hmac.compare_digest(row[0], token_hash):
                return False
            cursor.execute(
                """
                DELETE FROM research_workflow_operations
                WHERE session_id = %s
                """,
                (session_id,),
            )
            cursor.execute(
                "DELETE FROM research_audit_events WHERE session_id = %s",
                (session_id,),
            )
            cursor.execute("DELETE FROM research_sessions WHERE id = %s", (session_id,))
            return True

    def set_operation(
        self, session_id: str, status: str, detail: str, kind: str
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO research_workflow_operations(
                    session_id, status, detail, kind, updated_at
                ) VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT(session_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    detail = EXCLUDED.detail,
                    kind = EXCLUDED.kind,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW()
                """,
                (session_id, status, detail, kind),
            )

    def operation(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, detail, kind, lease_owner, lease_expires_at,
                       attempt, updated_at
                FROM research_workflow_operations WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return {"status": "idle", "detail": "", "kind": "generation"}
        return {
            "status": row[0],
            "detail": row[1],
            "kind": row[2],
            "lease_owner": row[3],
            "lease_expires_at": row[4].isoformat() if row[4] else None,
            "attempt": row[5],
            "updated_at": row[6].isoformat(),
        }

    def claim_operation(
        self,
        session_id: str,
        owner: str,
        *,
        detail: str,
        lease_seconds: int = 300,
    ) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_workflow_operations
                SET status = 'running', detail = %s, lease_owner = %s,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    attempt = attempt + 1, updated_at = NOW()
                WHERE session_id = %s
                  AND (
                    status = 'queued'
                    OR (
                        status = 'running'
                        AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
                    )
                  )
                """,
                (detail, owner, lease_seconds, session_id),
            )
            return cursor.rowcount == 1

    def renew_operation(
        self,
        session_id: str,
        owner: str,
        *,
        detail: str | None = None,
        lease_seconds: int = 300,
    ) -> bool:
        """Extend this worker's lease. See :meth:`ResearchLedger.renew_operation`."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_workflow_operations
                SET detail = COALESCE(%s, detail),
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE session_id = %s AND status = 'running' AND lease_owner = %s
                """,
                (detail, lease_seconds, session_id, owner),
            )
            return cursor.rowcount == 1

    def healthcheck(self) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)

    def requeue_expired_operation(self, session_id: str) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_workflow_operations
                SET status = 'queued', detail = 'Recovering interrupted work.',
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = NOW()
                WHERE session_id = %s AND status = 'running'
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < NOW()
                """,
                (session_id,),
            )
            return cursor.rowcount == 1

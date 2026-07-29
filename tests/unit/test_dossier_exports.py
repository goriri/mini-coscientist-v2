from io import BytesIO
from zipfile import ZipFile

from coscientist.dossier import render_docx, render_pdf

DOSSIER = """# Co-Scientist Research Dossier

**Question:** Does the intervention improve the measured outcome?

## Executive synthesis

- Candidate 1 remains testable.
- Independent replication is required.

## Decision audit

| Candidate | Decision |
| --- | --- |
| Candidate 1 | Conditional advance |
"""


def test_pdf_export_is_a_valid_pdf():
    exported = render_pdf(DOSSIER)

    assert exported.startswith(b"%PDF-")
    assert len(exported) > 1000


def test_docx_export_is_google_docs_compatible():
    exported = render_docx(DOSSIER)

    assert exported.startswith(b"PK")
    with ZipFile(BytesIO(exported)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Co-Scientist Research Dossier" in document_xml
    assert "Candidate 1 remains testable" in document_xml

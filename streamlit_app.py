# import base64
import io
import logging
import zipfile
from pathlib import Path

import streamlit as st
# import streamlit.components.v1 as components

import manage_docs
from ingestion.extractors import extract_docx, extract_txt
from ingestion.qdrant_manager import ConfigurationError, connect_qdrant


logger = logging.getLogger(__name__)

DOCS_FOLDER = manage_docs.DOCS_FOLDER
SUPPORTED_EXTENSIONS = manage_docs.SUPPORTED_EXTENSIONS
FLASH_KEY = "_streamlit_flash"

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


st.set_page_config(page_title="Document Manager", layout="wide")


def main():
    st.title("Document Manager")
    st.caption("Admin UI for the existing RAG ingestion pipeline. `docs/` remains the source of truth.")

    upload_tab, list_tab = st.tabs(["Upload Documents", "Document List"])

    with upload_tab:
        _render_upload_tab()

    with list_tab:
        _render_document_list_tab()


def _render_upload_tab():
    st.subheader("Upload Documents")
    st.write("Supported file types: PDF, DOCX, TXT")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Upload and Ingest", type="primary")
    if process_clicked:
        if not uploaded_files:
            st.info("Choose one or more files before uploading.")
        else:
            try:
                saved_paths, skipped_existing, save_failed = _save_uploaded_files(uploaded_files)
            except Exception as exc:
                logger.exception("Failed while saving uploaded files")
                st.error(f"Upload failed: {exc}")
                return

            if saved_paths:
                st.success("Saved to docs/: " + ", ".join(path.name for path in saved_paths))
            if skipped_existing:
                st.warning("Already present in docs/ and not overwritten: " + ", ".join(skipped_existing))
            if save_failed:
                st.error("Some files could not be saved.")
                for filename, reason in save_failed.items():
                    st.write(f"- {filename}: {reason}")

            try:
                client = connect_qdrant()
                added, duplicates, failed = manage_docs.add_documents(client, DOCS_FOLDER)
            except ConfigurationError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                logger.exception("Ingestion failed")
                st.error(f"Ingestion failed: {exc}")
                return

            relevant_names = {file.name for file in uploaded_files}
            added, duplicates, failed = _filter_summary_results(
                added,
                duplicates,
                failed,
                relevant_names,
            )
            st.session_state[FLASH_KEY] = {
                "kind": "ingestion",
                "added": added,
                "duplicates": duplicates,
                "failed": failed,
            }
            st.rerun()

    _render_flash_message("ingestion")


def _render_document_list_tab():
    st.subheader("Document List")
    _render_flash_message("deletion")

    documents = _list_documents()
    st.write(f"{len(documents)} supported document(s) found in docs/.")

    if not documents:
        st.info("No supported documents are currently stored in docs/.")
        return

    header_columns = st.columns([4, 1, 1, 1])
    header_columns[0].markdown("**Document Name**")
    header_columns[1].markdown("**View**")
    header_columns[2].markdown("**Download**")
    header_columns[3].markdown("**Delete**")

    for document_path in documents:
        _render_document_row(document_path)

    st.divider()
    st.subheader("Download All Documents")
    zip_bytes = _build_documents_zip(documents)
    st.download_button(
        "Download All Documents",
        data=zip_bytes,
        file_name="documents.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Delete All Documents")
    confirm_delete_all = st.checkbox(
        "I understand this will delete all documents from Qdrant and docs/."
    )
    delete_all_disabled = not confirm_delete_all or not documents
    if st.button("Delete All Documents", disabled=delete_all_disabled, type="primary"):
        try:
            client = connect_qdrant()
            deleted, not_found, failed = manage_docs.delete_documents(
                client,
                [document.name for document in documents],
            )
        except ConfigurationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("Delete-all failed")
            st.error(f"Delete all failed: {exc}")
            return

        st.session_state[FLASH_KEY] = {
            "kind": "deletion",
            "deleted": deleted,
            "not_found": not_found,
            "failed": failed,
        }
        st.rerun()


def _render_document_row(document_path):
    columns = st.columns([4, 1, 1, 1])
    columns[0].write(document_path.name)

    view_clicked = columns[1].button("View", key=f"view_{document_path.name}")
    columns[2].download_button(
        "Download",
        data=document_path.read_bytes(),
        file_name=document_path.name,
        mime=_mime_type(document_path.suffix),
        key=f"download_{document_path.name}",
    )

    delete_clicked = columns[3].button(
        "Delete",
        key=f"delete_{document_path.name}",
        type="secondary",
    )

    if view_clicked:
        with st.container():
            st.markdown(f"**Preview: {document_path.name}**")
            _render_document_preview(document_path)

    if delete_clicked:
        try:
            client = connect_qdrant()
            deleted, not_found, failed = manage_docs.delete_documents(client, [document_path.name])
        except ConfigurationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("Delete failed")
            st.error(f"Delete failed: {exc}")
            return

        st.session_state[FLASH_KEY] = {
            "kind": "deletion",
            "deleted": deleted,
            "not_found": not_found,
            "failed": failed,
        }
        st.rerun()


def _render_document_preview(document_path):
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        pdf_bytes = document_path.read_bytes()
        st.pdf(pdf_bytes)
        return

    if suffix == ".docx":
        records = extract_docx(document_path)
        text = "\n\n".join(record["text"] for record in records)
        st.text_area("Document content", value=text, height=400, disabled=True)
        return

    if suffix == ".txt":
        records = extract_txt(document_path)
        text = "\n\n".join(record["text"] for record in records)
        st.text_area("Document content", value=text, height=400, disabled=True)
        return

    st.warning(f"Unsupported file type: {suffix}")


def _save_uploaded_files(uploaded_files):
    saved_paths = []
    skipped_existing = []
    failed = {}

    for uploaded_file in uploaded_files:
        target_path = DOCS_FOLDER / uploaded_file.name
        try:
            if target_path.exists():
                skipped_existing.append(uploaded_file.name)
                continue

            target_path.write_bytes(uploaded_file.getvalue())
            saved_paths.append(target_path)
        except Exception as exc:
            logger.exception("Failed to save uploaded file %s", uploaded_file.name)
            failed[uploaded_file.name] = str(exc)

    return saved_paths, skipped_existing, failed


def _list_documents():
    if not DOCS_FOLDER.exists():
        return []

    documents = [
        path
        for path in DOCS_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(documents, key=lambda path: path.name.lower())


def _build_documents_zip(documents):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for document_path in documents:
            archive.writestr(document_path.name, document_path.read_bytes())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _filter_summary_results(added, duplicates, failed, names):
    filtered_added = [name for name in added if name in names]
    filtered_duplicates = [name for name in duplicates if name in names]
    filtered_failed = {name: reason for name, reason in failed.items() if name in names}
    return filtered_added, filtered_duplicates, filtered_failed


def _render_ingestion_summary(added, duplicates, failed):
    st.subheader("Ingestion Summary")
    _render_file_section("Added Successfully", added, marker="+")
    _render_file_section("Skipped (Duplicate)", duplicates, marker="-")
    _render_failed_section(failed)
    total = len(added) + len(duplicates) + len(failed)
    st.write(f"Total Files: {total}")
    st.write(f"Added: {len(added)}")
    st.write(f"Duplicates: {len(duplicates)}")
    st.write(f"Failed: {len(failed)}")


def _render_deletion_summary(deleted, not_found, failed):
    st.subheader("Deletion Summary")
    _render_file_section("Deleted Successfully", deleted, marker="+")
    _render_file_section("Not Found", not_found, marker="-")
    _render_failed_section(failed)
    total = len(deleted) + len(not_found) + len(failed)
    st.write(f"Total Files: {total}")
    st.write(f"Deleted: {len(deleted)}")
    st.write(f"Not Found: {len(not_found)}")
    st.write(f"Failed: {len(failed)}")


def _render_file_section(title, filenames, marker):
    st.markdown(f"**{title}:**")
    if not filenames:
        st.write("None")
        return
    for filename in filenames:
        st.write(f"{marker} {filename}")


def _render_failed_section(failed):
    st.markdown("**Failed:**")
    if not failed:
        st.write("None")
        return

    for filename in failed:
        st.write(f"x {filename}")

    st.markdown("**Reason:**")
    for filename, reason in failed.items():
        st.write(f"{filename}: {reason}")


def _render_flash_message(expected_kind):
    flash = st.session_state.get(FLASH_KEY)
    if not flash or flash.get("kind") != expected_kind:
        return

    if expected_kind == "ingestion":
        _render_ingestion_summary(
            flash["added"],
            flash["duplicates"],
            flash["failed"],
        )
    elif expected_kind == "deletion":
        _render_deletion_summary(
            flash["deleted"],
            flash["not_found"],
            flash["failed"],
        )

    if st.button("OK", key=f"ok_{expected_kind}"):
        st.session_state.pop(FLASH_KEY, None)
        st.rerun()


def _mime_type(suffix):
    return MIME_TYPES.get(suffix.lower(), "application/octet-stream")


if __name__ == "__main__":
    main()

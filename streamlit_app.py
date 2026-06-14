# import base64
import io
import logging
import zipfile
from pathlib import Path

import streamlit as st
# import streamlit.components.v1 as components

import manage_docs
from chat import cache_manager, conversation_manager, message_manager
from ingestion.extractors import extract_docx, extract_txt
from ingestion.qdrant_manager import ConfigurationError, connect_qdrant
from retrieval.agent import run_agent
from retrieval.company_tool import _get_reranker


logger = logging.getLogger(__name__)

DOCS_FOLDER = manage_docs.DOCS_FOLDER
SUPPORTED_EXTENSIONS = manage_docs.SUPPORTED_EXTENSIONS
FLASH_KEY = "_streamlit_flash"
SELECTED_CONVERSATION_KEY = "selected_conversation_id"
SHOW_NEW_CHAT_FORM_KEY = "show_new_chat_form"

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


st.set_page_config(page_title="RAG Assistant", layout="wide")

@st.cache_resource
def warmup_reranker():
    """
    Load the BGE reranker once and keep it alive for the lifetime
    of the Streamlit process.
    """
    return _get_reranker()

@st.cache_resource
def warmup_qdrant():
    return connect_qdrant()

def main():
    warmup_reranker()
    warmup_qdrant()
    st.title("RAG Assistant")
    st.caption("Manage documents and chat with the company knowledge assistant.")

    _render_chat_sidebar()

    document_tab, chat_tab = st.tabs(["Document Management", "Chat"])

    with document_tab:
        _render_document_management_section()

    with chat_tab:
        _render_chat_section()


def _render_document_management_section():
    st.header("Document Management")
    st.caption("Admin UI for the existing RAG ingestion pipeline. `docs/` remains the source of truth.")

    upload_tab, list_tab = st.tabs(["Upload Documents", "Document List"])

    with upload_tab:
        _render_upload_tab()

    with list_tab:
        _render_document_list_tab()


def _render_chat_sidebar():
    st.sidebar.title("Chat")

    if st.sidebar.button("New Chat", use_container_width=True):
        st.session_state[SHOW_NEW_CHAT_FORM_KEY] = True

    if st.session_state.get(SHOW_NEW_CHAT_FORM_KEY):
        with st.sidebar.form("new_chat_form", clear_on_submit=True):
            title = st.text_input("Conversation title")
            create_clicked = st.form_submit_button("Create Chat")

            if create_clicked:
                if not title.strip():
                    st.warning("Enter a title before creating a chat.")
                else:
                    try:
                        conversation = conversation_manager.create_conversation(
                            title.strip()
                        )
                        _select_conversation(conversation["conversation_id"])
                        st.session_state[SHOW_NEW_CHAT_FORM_KEY] = False
                        st.rerun()
                    except Exception:
                        logger.exception("Failed to create conversation")
                        st.sidebar.error("Could not create the chat. Please try again.")

    st.sidebar.divider()
    st.sidebar.subheader("Conversations")

    try:
        conversations = conversation_manager.list_conversations()
    except Exception:
        logger.exception("Failed to list conversations")
        st.sidebar.error("Could not load conversations.")
        return

    if not conversations:
        st.sidebar.info("No conversations yet.")
        return

    selected_conversation_id = st.session_state.get(SELECTED_CONVERSATION_KEY)

    for conversation in conversations:
        conversation_id = conversation.get("conversation_id")
        title = conversation.get("title") or "Untitled chat"
        title_label = title
        if conversation_id == selected_conversation_id:
            title_label = f"> {title}"

        title_column, delete_column = st.sidebar.columns([4, 1])

        if title_column.button(
            title_label,
            key=f"select_{conversation_id}",
            use_container_width=True,
        ):
            _select_conversation(conversation_id)
            st.rerun()

        if delete_column.button(
            "x",
            key=f"delete_conversation_{conversation_id}",
            help=f"Delete {title}",
        ):
            _delete_conversation(conversation_id)
            st.rerun()


def _render_chat_section():
    st.header("Chat")

    conversation_id = st.session_state.get(SELECTED_CONVERSATION_KEY)
    if not conversation_id:
        st.info("Create or select a conversation from the sidebar to start chatting.")
        return

    try:
        conversation = conversation_manager.load_conversation_metadata(conversation_id)
    except Exception:
        logger.exception("Failed to load conversation metadata")
        st.error("Could not load the selected conversation.")
        return

    if conversation is None:
        st.warning("The selected conversation no longer exists.")
        st.session_state.pop(SELECTED_CONVERSATION_KEY, None)
        return

    st.subheader(conversation.get("title") or "Untitled chat")

    try:
        messages = message_manager.load_messages(conversation_id)
    except Exception:
        logger.exception("Failed to load messages")
        st.error("Could not load this conversation's messages.")
        return

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role not in {"user", "assistant"} or not content:
            continue

        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Ask a question")
    if not prompt:
        return

    _handle_chat_prompt(conversation_id, prompt)
    st.rerun()


def _handle_chat_prompt(conversation_id, prompt):
    try:
        message_manager.insert_message(conversation_id, "user", prompt)
        cache_manager.append_message(conversation_id, "user", prompt)
    except Exception:
        logger.exception("Failed to save user message")
        st.error("Could not save your message. Please try again.")
        return

    try:
        with st.spinner("Thinking..."):
            answer = run_agent(prompt, conversation_id)
    except Exception:
        logger.exception("Failed to generate assistant response")
        st.error("I could not generate a response. Please try again.")
        return

    try:
        message_manager.insert_message(conversation_id, "assistant", answer)
        cache_manager.append_message(conversation_id, "assistant", answer)
    except Exception:
        logger.exception("Failed to save assistant response")
        st.error("The response was generated, but could not be saved.")


def _select_conversation(conversation_id):
    st.session_state[SELECTED_CONVERSATION_KEY] = conversation_id
    cache_manager.rebuild_cache(conversation_id)


def _delete_conversation(conversation_id):
    try:
        conversation_manager.delete_conversation(conversation_id)
        message_manager.delete_messages(conversation_id)
        cache_manager.clear_cache(conversation_id)
    except Exception:
        logger.exception("Failed to delete conversation")
        st.sidebar.error("Could not delete the chat. Please try again.")
        return

    if st.session_state.get(SELECTED_CONVERSATION_KEY) == conversation_id:
        st.session_state.pop(SELECTED_CONVERSATION_KEY, None)


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

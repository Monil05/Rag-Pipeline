import logging
import sys
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
DOCS_FOLDER = Path("docs")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class CliError(RuntimeError):
    pass


def main():
    try:
        command = _parse_command(sys.argv)
        if command["name"] == "add":
            folder = Path(command["folder"])
            _validate_folder(folder)
        if command["name"] == "rebuild":
            _validate_folder(DOCS_FOLDER)

        from ingestion.qdrant_manager import (
            connect_qdrant,
            create_collection_if_not_exists,
            delete_collection,
        )

        client = connect_qdrant()
        create_collection_if_not_exists(client)

        if command["name"] == "add":
            added, duplicates, failed = add_documents(client, folder)
            print_ingestion_summary(added, duplicates, failed)
            return

        if command["name"] == "delete":
            deleted, not_found, failed = delete_documents(client, command["filenames"])
            print_deletion_summary(deleted, not_found, failed)
            return

        if command["name"] == "rebuild":
            delete_collection(client)
            create_collection_if_not_exists(client)
            added, duplicates, failed = add_documents(client, DOCS_FOLDER)
            print_ingestion_summary(added, duplicates, failed)
            return
    except CliError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Command failed")
        print(f"Error: {exc}")
        sys.exit(1)


def add_documents(client, folder):
    from ingestion.qdrant_manager import document_exists, insert_chunks

    _validate_folder(folder)

    added = []
    duplicates = []
    failed = {}

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            document_name = file_path.name
            if document_exists(client, document_name):
                duplicates.append(document_name)
                continue

            chunks = _prepare_chunks(file_path)
            insert_chunks(client, chunks)
            added.append(document_name)
        except Exception as exc:
            logger.exception("Failed to process %s", file_path.name)
            failed[file_path.name] = str(exc)

    return added, duplicates, failed


def delete_documents(client, filenames):
    from ingestion.qdrant_manager import delete_document, document_exists

    deleted = []
    not_found = []
    failed = {}

    for filename in filenames:
        try:
            file_path = DOCS_FOLDER / filename
            file_exists = file_path.exists()
            qdrant_exists = document_exists(client, filename)

            if not file_exists and not qdrant_exists:
                not_found.append(filename)
                continue

            if qdrant_exists:
                delete_document(client, filename)

            if file_exists:
                file_path.unlink()

            deleted.append(filename)
        except Exception as exc:
            logger.exception("Failed to delete %s", filename)
            failed[filename] = str(exc)

    return deleted, not_found, failed


def print_ingestion_summary(added, duplicates, failed):
    print("\n========== INGESTION SUMMARY ==========\n")
    _print_file_section("Added Successfully:", added, "\u2713")
    _print_file_section("Skipped (Duplicate):", duplicates, "\u2022")
    _print_failed_section(failed)
    print(f"Total Files: {len(added) + len(duplicates) + len(failed)}")
    print(f"Added: {len(added)}")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Failed: {len(failed)}")


def print_deletion_summary(deleted, not_found, failed):
    print("\n========== DELETION SUMMARY ==========\n")
    _print_file_section("Deleted Successfully:", deleted, "\u2713")
    _print_file_section("Not Found:", not_found, "\u2022")
    _print_failed_section(failed)
    print(f"Total Files: {len(deleted) + len(not_found) + len(failed)}")
    print(f"Deleted: {len(deleted)}")
    print(f"Not Found: {len(not_found)}")
    print(f"Failed: {len(failed)}")


def _prepare_chunks(file_path):
    from ingestion.chunking import chunk_document
    from ingestion.embeddings import get_embedding

    source_type = file_path.suffix.lower().lstrip(".")
    page_records = _extract_file(file_path)
    chunks = chunk_document(
        page_records=page_records,
        document_name=file_path.name,
        source_type=source_type,
    )

    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["text"])

    return chunks


def _extract_file(file_path):
    from ingestion.extractors import extract_docx, extract_pdf, extract_txt

    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return extract_pdf(file_path)
    if extension == ".docx":
        return extract_docx(file_path)
    if extension == ".txt":
        return extract_txt(file_path)
    raise RuntimeError(f"Unsupported file type: {extension}")


def _parse_command(args):
    usage = (
        "Usage:\n"
        "  python manage_docs.py add docs/\n"
        "  python manage_docs.py delete leave_policy.pdf [reimbursement.docx ...]\n"
        "  python manage_docs.py rebuild"
    )

    if len(args) < 2:
        raise CliError(usage)

    command = args[1].lower()
    if command == "add" and len(args) == 3:
        return {"name": "add", "folder": args[2]}
    if command == "delete" and len(args) >= 3:
        return {"name": "delete", "filenames": args[2:]}
    if command == "rebuild" and len(args) == 2:
        return {"name": "rebuild"}

    raise CliError(usage)


def _validate_folder(folder):
    if not folder.exists() or not folder.is_dir():
        raise CliError(f"Folder does not exist: {folder}")


def _print_file_section(title, filenames, marker):
    print(title)
    if filenames:
        for filename in filenames:
            print(f"{marker} {filename}")
    else:
        print("None")
    print()


def _print_failed_section(failed):
    print("Failed:")
    if not failed:
        print("None\n")
        return

    for filename in failed:
        print(f"\u2717 {filename}")

    print("\nReason:")
    for filename, reason in failed.items():
        print(f"{filename}: {reason}")
    print()


if __name__ == "__main__":
    main()

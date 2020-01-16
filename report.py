import sys
import os
import zipfile
import webbrowser
assert sys.version_info >= (3,7), "messenger_stats requires python version 3.7 or newer."

if __name__ == "__main__":
    data_archive: zipfile.ZipFile
    while True:
        try:
            archive_path = os.path.abspath(input("Path to downloaded Facebook data in ZIP format: ").strip())
            if os.path.isdir(archive_path):
                sys.stderr.write(f"[ERROR] {archive_path} is a folder. Do not extract the archive.\n")
                continue
            data_archive = zipfile.ZipFile(archive_path, allowZip64=True, mode="r")
            break
        except FileNotFoundError:
            sys.stderr.write(f"[ERROR] File {archive_path} not found.\n")
        except zipfile.BadZipFile:
            sys.stderr.write(f"[ERROR] {archive_path} is not a valid ZIP file\n")
        pass

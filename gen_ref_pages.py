"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()  # hold navigation parts

for path in sorted(
    Path("housing").rglob("*.py")
):  # finds all directories and files matching pattern
    module_path = path.relative_to(".").with_suffix("")  # removes the .py suffix
    doc_path = path.relative_to(".").with_suffix(".md")  # pages will use .md suffix
    full_doc_path = Path(
        "reference", doc_path
    )  # generated pages are under reference dir

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  # remove __init__.py, skip __main__.py
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()  # adds navigation part

    with mkdocs_gen_files.open(
        full_doc_path, "w"
    ) as fd:  # adds file without explicitly storing it
        identifier = ".".join(parts)  # python module identifier
        print("::: " + identifier, file=fd)  # write autodoc identifier

    mkdocs_gen_files.set_edit_path(
        full_doc_path, path
    )  # sets edit path from doc page to python code

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())

import pathlib
import setuptools
HERE = pathlib.Path(__file__).parent.resolve()
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
setuptools.setup(
    name="vision",
    version="1.0.0",
    description="Personal Vision Projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="luke bennett",
    packages=setuptools.find_namespace_packages(include=["vision_projects.*"]),
    python_requires=">=3.5, <4"
)

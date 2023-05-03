from setuptools import setup, find_packages

setup(
    name="mk8dx_digit_ocr",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
        ],
    },
    entry_points={
        # コマンドラインツールを指定
        "console_scripts": [
            "mk8dx-digit-ocr = mk8dx_digit_ocr.digit_ocr:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    author="furaga",
    author_email="furaga.fukahori@gmail.com",
    description="7 segment digit OCR for Mario Kart 8 Deluxe",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/furaga/mk8dx-digit-ocr",
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
)

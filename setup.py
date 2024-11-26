from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setup(
	name="pylcg",
	version="1.0.2",
	author="acidvegas",
	author_email="acid.vegas@acid.vegas",
	description="Linear Congruential Generator for IP Sharding",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/acidvegas/pylcg",
	project_urls={
		"Bug Tracker": "https://github.com/acidvegas/pylcg/issues",
		"Documentation": "https://github.com/acidvegas/pylcg#readme",
		"Source Code": "https://github.com/acidvegas/pylcg",
	},
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: ISC License (ISCL)",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Topic :: Internet",
		"Topic :: Security",
		"Topic :: Software Development :: Libraries :: Python Modules",
	],
	packages=find_packages(),
	python_requires=">=3.6",
	entry_points={
		'console_scripts': [
			'pylcg=pylcg.cli:main',
		],
	},
)

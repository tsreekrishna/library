from setuptools import setup, find_packages

setup(name="package", 
      version="0.1", 
      long_description="This package has all the files related to crop classification", 
      author="Krishna",
      author_email="krishna@wadhawniai.org",
      url="https://github.com/tsreekrishna/library".
      packages=["package"], 
      install_requires=["scikit-learn", "pandas", "seaborn", "xgboost", "pickle", "copy"]
      )
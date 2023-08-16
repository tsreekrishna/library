from setuptools import setup, find_packages

setup(name="crop_classification", 
      version="0.0.1", 
      description="This package has all the files related to crop classification", 
      author="T Sree Krishna",
      author_email="krishna@wadhawniai.org",
      url="https://github.com/tsreekrishna/library",
      packages=find_packages(),
      python_requires = ">=3.8",
      install_requires=["scikit-learn", "pandas", "xgboost"],
      include_package_data=True)
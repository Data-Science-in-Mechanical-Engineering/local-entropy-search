from setuptools import setup, find_packages

# requirements = (
#   'numpy>=1.18.0',
#   'tensorflow>=2.2.0',
#   'tensorflow-probability>=0.9.0',
#   'gpflow>=2.0.3',
# )

setup(name='local_bo',
      version='0.1',
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      packages=find_packages(),
      python_requires='>=3.6')
    #   install_requires=requirements)
from setuptools import setup, find_packages

setup(
    name='py2flows',
    version='0.1',
    zip_safe=False,
    packages=find_packages(),
    url='https://github.com/LayneInNL/py2flows',
    license='APACHE LICENSE, VERSION 2.0',
    author='Layne Liu',
    author_email='layne.liu@outlook.com',
    description='A control flow generator for Python',
    entry_points={
        'console_scripts': [
            'py2flows = py2flows.main:main'
        ]
    }

)

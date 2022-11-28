from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name = 'MCG',
        version = '0.1',
        license = 'MIT',
        packages = find_packages(),
        install_requires = [
            'numpy',
            'pandas',
            'matplotlib',
            'scikit-learn',
            'pillow',
            'ipykernel',
            'jupyter',
            'tensorflow-macos; sys_platform == "darwin"',
            'tensorflow; sys_platform in ["win32", "cygwin"]'
        ]
    )

from setuptools import find_packages, setup
from glob import glob

package_name = 'haar_upperbody'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Menyertakan file XML di folder cascades ke instalasi package
        ('share/' + package_name + '/cascades', glob('cascades/*.xml')),
        # File package.xml jika ada
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pcku',
    maintainer_email='apriantowpj@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
     'console_scripts': [
         'detect_upperbody = haar_upperbody.detect_upperbody:main',
   	 ],
    },
)

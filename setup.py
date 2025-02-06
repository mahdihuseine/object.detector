from setuptools import setup, find_packages

setup(
    name="object.detector",  # نام کتابخانه شما
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'ultralytics',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'object-detector=object.detector.detect:main',  # نام تابع main که در آن کارهای اصلی انجام می‌شود
        ],
    },
    include_package_data=True,
    description="A library for car and license plate detection",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/object.detector",  # آدرس گیت‌هاب یا صفحه پروژه شما
    author="Your Name",
    author_email="your-email@example.com",
    license="MIT",
)

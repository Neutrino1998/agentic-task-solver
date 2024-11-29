from setuptools import setup, find_packages

setup(
    name='agentic_task_solver',  # 包名
    version='0.1.0',
    packages=find_packages(),  # 自动发现所有的包
    install_requires=[  # 如果有其他依赖，列在这里
        # 'some_package',
    ],
    # 可以添加其他设置，比如作者信息等
    author='同温层',
    author_email='1998neutrino@gmail.com',
    description='A multi-agent system designed to collectively solve a given task based on langchain & langgraph.',
)

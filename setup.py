from setuptools import setup, find_packages

def load_requirements(filename="requirements.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()

setup(
    name='agentic_task_solver',  
    version='0.1.0',
    packages=find_packages(),  
    install_requires=load_requirements(), 
    author='同温层',
    author_email='1998neutrino@gmail.com',
    description='A multi-agent system designed to collectively solve a given task based on langchain & langgraph.',
)

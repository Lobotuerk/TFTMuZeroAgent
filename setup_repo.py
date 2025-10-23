#!/usr/bin/env python3
"""
TFT Set 4 Gym Repository Setup Script

This script helps you create the TFT-Set4-Gym repository and set it up as a submodule.
Run this script to automate the repository creation process.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr and check:
        print(f"Error: {result.stderr}")
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


def setup_tft_gym_repo():
    """Set up the TFT-Set4-Gym repository."""
    
    print("🚀 Setting up TFT-Set4-Gym Repository")
    print("=" * 50)
    
    # Create the new repository directory
    repo_path = Path("../TFT-Set4-Gym")
    if repo_path.exists():
        print(f"⚠️  Directory {repo_path} already exists!")
        response = input("Do you want to remove it and start fresh? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(repo_path)
        else:
            print("Aborting setup.")
            return
    
    print(f"📁 Creating directory: {repo_path}")
    repo_path.mkdir(parents=True, exist_ok=True)
    
    # Create the package directory
    package_path = repo_path / "tft_set4_gym"
    package_path.mkdir(exist_ok=True)
    
    print("📋 Copying Simulator files...")
    
    # Copy all Simulator files
    simulator_path = Path("Simulator")
    for item in simulator_path.rglob("*"):
        if item.is_file() and not item.name.endswith(".pyc"):
            # Calculate relative path and destination
            rel_path = item.relative_to(simulator_path)
            dest_path = package_path / rel_path
            
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"  Copying: {item} -> {dest_path}")
            shutil.copy2(item, dest_path)
    
    # Copy setup files
    setup_files = [
        ("setup.py", repo_path / "setup.py"),
        ("TFT_SET4_README.md", repo_path / "README.md"),
        ("tft_requirements.txt", repo_path / "requirements.txt"),
        ("tft_config.py", package_path / "config.py"),
        ("tft_set4_gym_init.py", package_path / "__init__.py"),
        ("tft_wrappers.py", package_path / "wrappers.py"),
    ]
    
    print("📋 Copying setup files...")
    for src, dest in setup_files:
        if Path(src).exists():
            print(f"  Copying: {src} -> {dest}")
            shutil.copy2(src, dest)
        else:
            print(f"  ⚠️  Skipping missing file: {src}")
    
    # Update imports in the copied files
    print("🔧 Updating import statements...")
    update_imports(package_path)
    
    # Create additional files
    create_additional_files(repo_path, package_path)
    
    print("✅ Repository structure created successfully!")
    print(f"📁 Repository location: {repo_path.absolute()}")
    
    return repo_path


def update_imports(package_path):
    """Update import statements in the copied files."""
    
    # Files that need import updates
    files_to_update = [
        "tft_simulator.py",
        "config.py", 
        "player.py",
        "observation.py",
        "minion.py",
        "__init__.py",
        "wrappers.py"
    ]
    
    for filename in files_to_update:
        file_path = package_path / filename
        if not file_path.exists():
            continue
            
        print(f"  Updating imports in: {filename}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update imports
        if filename == "tft_simulator.py":
            content = content.replace("import config", "from . import config")
            content = content.replace("from Simulator import pool", "from . import pool")
            content = content.replace("from Simulator.player import Player as player_class", "from .player import Player as player_class")
            content = content.replace("from Simulator.step_function import Step_Function", "from .step_function import Step_Function")
            content = content.replace("from Simulator.game_round import Game_Round", "from .game_round import Game_Round")
            content = content.replace("from Simulator.observation import Observation", "from .observation import Observation")
        
        elif filename == "config.py":
            content = content.replace("import config", "# Removed main config import")
            content = content.replace("from Simulator.item_stats import uncraftable_items", "from .item_stats import uncraftable_items")
        
        elif filename in ["player.py", "observation.py", "minion.py"]:
            content = content.replace("from config import DEBUG", "from .config import DEBUG")
            content = content.replace("import config", "from . import config")
        
        elif filename == "__init__.py":
            content = content.replace("from .tft_simulator import parallel_env, env", "from .tft_simulator import parallel_env, env")
            content = content.replace("from .wrappers import TFTSingleAgentWrapper", "from .wrappers import TFTSingleAgentWrapper")
        
        elif filename == "wrappers.py":
            content = content.replace("from .tft_simulator import parallel_env", "from .tft_simulator import parallel_env")
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def create_additional_files(repo_path, package_path):
    """Create additional files needed for the repository."""
    
    # Create LICENSE file
    license_content = """MIT License

Copyright (c) 2025 Lobotuerk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open(repo_path / "LICENSE", 'w') as f:
        f.write(license_content)
    
    # Create .gitignore
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"""
    
    with open(repo_path / ".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    # Create demo script
    demo_content = """#!/usr/bin/env python3
\"\"\"
Demo script for TFT Set 4 Gymnasium Environment.
\"\"\"

import numpy as np
from tft_set4_gym import parallel_env


def main():
    \"\"\"Run a basic demo of the TFT environment.\"\"\"
    print("🎮 TFT Set 4 Gym Demo")
    print("=" * 30)
    
    # Create environment
    env = parallel_env()
    
    # Reset environment
    observations, infos = env.reset()
    print(f"✅ Environment reset with {len(env.agents)} agents")
    
    # Run for a few steps
    for step in range(10):
        # Sample random actions for all agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print step info
        active_agents = len([a for a in env.agents if not (terminations[a] or truncations[a])])
        total_reward = sum(rewards.values())
        
        print(f"Step {step+1}: {active_agents} active agents, total reward: {total_reward:.1f}")
        
        # Remove terminated/truncated agents
        env.agents = [agent for agent in env.agents 
                      if not (terminations[agent] or truncations[agent])]
        
        if len(env.agents) <= 1:
            print("Game ended!")
            break
    
    env.close()
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()
"""
    
    with open(package_path / "demo.py", 'w') as f:
        f.write(demo_content)


def initialize_git_repo(repo_path):
    """Initialize git repository and create initial commit."""
    
    print("\n🔧 Initializing Git repository...")
    
    # Initialize git repo
    run_command("git init", cwd=repo_path)
    
    # Add all files
    run_command("git add .", cwd=repo_path)
    
    # Create initial commit
    run_command(['git', 'commit', '-m', 'Initial commit: TFT Set 4 Gymnasium Environment'], cwd=repo_path)
    
    print("✅ Git repository initialized!")


def setup_github_remote(repo_path):
    """Set up GitHub remote repository."""
    
    print("\n🌐 Setting up GitHub remote...")
    
    # Add remote
    remote_url = "https://github.com/Lobotuerk/TFT-Set4-Gym.git"
    run_command(['git', 'remote', 'add', 'origin', remote_url], cwd=repo_path)
    
    print(f"✅ Remote added: {remote_url}")
    print("\n📋 Next steps:")
    print("1. Create the repository on GitHub: https://github.com/new")
    print("2. Repository name: TFT-Set4-Gym")
    print("3. Make it public")
    print("4. Don't initialize with README (we already have one)")
    print(f"5. Run: cd {repo_path.name} && git push -u origin main")


def add_as_submodule():
    """Add the new repository as a submodule to the current project."""
    
    print("\n📦 Adding as submodule...")
    
    # Remove existing Simulator directory
    if Path("Simulator").exists():
        print("🗑️  Removing existing Simulator directory...")
        shutil.rmtree("Simulator")
    
    # Add submodule
    submodule_url = "https://github.com/Lobotuerk/TFT-Set4-Gym.git"
    run_command(['git', 'submodule', 'add', submodule_url, 'Simulator'])
    
    # Update .gitmodules to point to the package directory
    gitmodules_content = f"""[submodule "Simulator"]
	path = Simulator
	url = {submodule_url}
"""
    
    with open(".gitmodules", 'w') as f:
        f.write(gitmodules_content)
    
    print("✅ Submodule added!")
    print("\n📋 Update your imports:")
    print("Replace: from Simulator.tft_simulator import parallel_env")
    print("With:    from Simulator.tft_set4_gym import parallel_env")


def main():
    """Main setup function."""
    
    print("🚀 TFT-Set4-Gym Repository Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("Simulator").exists():
        print("❌ Error: Simulator directory not found!")
        print("Please run this script from the TFTMuZeroAgent directory.")
        sys.exit(1)
    
    try:
        # Step 1: Create repository structure
        repo_path = setup_tft_gym_repo()
        
        # Step 2: Initialize Git
        initialize_git_repo(repo_path)
        
        # Step 3: Setup GitHub remote
        setup_github_remote(repo_path)
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Manual steps remaining:")
        print("1. Create GitHub repository: https://github.com/new")
        print("2. Repository name: TFT-Set4-Gym")
        print("3. Push the code:")
        print("   cd ../TFT-Set4-Gym")
        print("   git push -u origin main")
        print("4. Come back to this directory and run:")
        print("   python setup_repo.py --add-submodule")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if "--add-submodule" in sys.argv:
        add_as_submodule()
    else:
        main()
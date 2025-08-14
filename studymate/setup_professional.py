#!/usr/bin/env python3
"""
StudyMate Professional Setup Script
Installs dependencies and sets up the environment for the professional RAG system
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Install torch first (for better compatibility)
    if not run_command("pip install torch>=2.2.0", "Installing PyTorch"):
        print("⚠️ PyTorch installation failed. Continuing with other packages...")
    
    # Install main requirements
    if not run_command("pip install -r requirements.txt", "Installing main requirements"):
        return False
    
    # Install spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model", check=False):
        print("⚠️ spaCy model installation failed. You can install it later with: python -m spacy download en_core_web_sm")
    
    return True

def setup_environment():
    """Set up environment configuration"""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_template = Path("env_template.txt")
    
    if not env_file.exists() and env_template.exists():
        # Copy template to .env
        with open(env_template, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("📝 Please edit .env file to add your IBM Watson credentials")
    else:
        print("ℹ️ Environment file already exists or template not found")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "data/uploads",
        "data/advanced_indexes",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print("\n🌐 Installing frontend dependencies...")
    
    frontend_dir = Path("../studymate-frontend")
    if not frontend_dir.exists():
        print("⚠️ Frontend directory not found. Skipping frontend setup.")
        return True
    
    os.chdir(frontend_dir)
    
    # Check if Node.js is installed
    if not run_command("node --version", "Checking Node.js", check=False):
        print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False
    
    # Install dependencies
    if not run_command("npm install", "Installing frontend dependencies"):
        return False
    
    os.chdir("..")
    return True

def download_models():
    """Download and cache ML models"""
    print("\n🤖 Downloading ML models...")
    
    # Create a simple script to download models
    download_script = '''
import os
os.environ['TRANSFORMERS_CACHE'] = './models/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models/sentence_transformers'

from sentence_transformers import SentenceTransformer
import nltk

print("Downloading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer model downloaded")

print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("✅ NLTK data downloaded")
'''
    
    with open("download_models.py", "w", encoding='utf-8') as f:
        f.write(download_script)
    
    if run_command("python download_models.py", "Downloading ML models"):
        os.remove("download_models.py")
        return True
    else:
        print("⚠️ Model download failed. Models will be downloaded on first use.")
        return False

def main():
    """Main setup function"""
    print("🚀 StudyMate Professional Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Download models (optional)
    download_models()
    
    # Install frontend dependencies
    # install_frontend_dependencies()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your IBM Watson credentials")
    print("2. Run the backend: python professional_server.py")
    print("3. Run the frontend: cd ../studymate-frontend && npm start")
    print("4. Open http://localhost:3000 in your browser")
    print("\n📚 Documentation: Check README.md for more details")
    print("🐛 Issues: Report bugs on GitHub")

if __name__ == "__main__":
    main()

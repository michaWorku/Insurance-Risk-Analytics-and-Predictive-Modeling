import pytest
import pandas as pd
from pathlib import Path
import os
import sys

# Add src to system path to allow imports from src.data_loader
# Assuming tests/unit is at project_root/tests/unit
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print("path:", str(Path(__file__).parent.parent.parent / 'src'))

from src.utils.data_loader import load_data

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Fixture to create a temporary directory for dummy data files."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    return data_dir

def test_load_data_csv_default_delimiter(dummy_data_dir):
    """Test loading a CSV with default comma delimiter."""
    file_path = dummy_data_dir / "test.csv"
    content = "col1,col2\n1,A\n2,B"
    file_path.write_text(content)
    
    df = load_data(file_path)
    assert not df.empty
    pd.testing.assert_frame_equal(df, pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']}))

def test_load_data_txt_pipe_delimiter_inferred(dummy_data_dir):
    """Test loading a TXT with pipe delimiter, inferred by function."""
    file_path = dummy_data_dir / "test_pipe.txt"
    content = "col1|col2\n1|A\n2|B"
    file_path.write_text(content)
    
    df = load_data(file_path)
    assert not df.empty
    pd.testing.assert_frame_equal(df, pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']}))

def test_load_data_txt_tab_delimiter_explicit(dummy_data_dir):
    """Test loading a TXT with tab delimiter, explicitly specified."""
    file_path = dummy_data_dir / "test_tab.txt"
    content = "col1\tcol2\n1\tA\n2\tB"
    file_path.write_text(content)
    
    df = load_data(file_path, delimiter='\t')
    assert not df.empty
    pd.testing.assert_frame_equal(df, pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']}))

def test_load_data_non_existent_file(dummy_data_dir, capsys):
    """Test loading a file that does not exist."""
    file_path = dummy_data_dir / "non_existent.csv"
    df = load_data(file_path)
    assert df.empty
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.out

def test_load_data_empty_file(dummy_data_dir, capsys):
    """Test loading an empty file."""
    file_path = dummy_data_dir / "empty.csv"
    file_path.write_text("")
    df = load_data(file_path)
    assert df.empty
    captured = capsys.readouterr()
    assert "Error: File 'empty.csv' is empty." in captured.out

def test_load_data_unsupported_file_type(dummy_data_dir, capsys):
    """Test loading an unsupported file type."""
    file_path = dummy_data_dir / "image.jpg"
    file_path.write_text("dummy content")
    df = load_data(file_path)
    assert df.empty
    captured = capsys.readouterr()
    assert "Error: Unsupported file type 'jpg'" in captured.out

def test_load_data_parser_error(dummy_data_dir, capsys):
    """Test loading a CSV file with incorrect delimiter leading to parsing error."""
    file_path = dummy_data_dir / "malformed.csv"
    content = "col1;col2\n1;A\n2;B" # Semicolon separated, but trying to read as comma
    file_path.write_text(content)
    
    df = load_data(file_path, delimiter=',') # Force comma delimiter
    assert df.empty
    captured = capsys.readouterr()
    assert "Error parsing data" in captured.out

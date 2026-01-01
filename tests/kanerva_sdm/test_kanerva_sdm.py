"""
Unit tests for Kanerva SDM.

(c) 2026 Simon Wong
"""

import pytest
import kanerva_sdm


class TestKanervaSDM:
    """Test suite for KanervaSDM class."""

    def test_initialization(self):
        """Test SDM initialization with valid parameters."""
        sdm = kanerva_sdm.KanervaSDM(
            address_dimension=100,
            memory_dimension=100,
            num_locations=10000,
            hamming_threshold=37,
            random_seed=42
        )
        assert sdm.address_dimension == 100
        assert sdm.memory_dimension == 100
        assert sdm.num_locations == 10000
        assert sdm.hamming_threshold == 37
        assert sdm.memory_count == 0

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError):
            kanerva_sdm.KanervaSDM(0, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            kanerva_sdm.KanervaSDM(256, -1, 1000, 103)
        
        with pytest.raises(ValueError):
            kanerva_sdm.KanervaSDM(256, 128, 0, 103)

    def test_invalid_threshold(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError):
            kanerva_sdm.KanervaSDM(256, 128, 1000, -1)

    def test_write_and_read(self):
        """Test basic write and read operations."""
        sdm = kanerva_sdm.KanervaSDM(
            address_dimension=256,
            memory_dimension=128,
            num_locations=1000,
            hamming_threshold=103,
            random_seed=42
        )
        
        address = [0] * 256
        memory = [1] * 128
        
        sdm.write(address, memory)
        assert sdm.memory_count == 1
        
        recalled = sdm.read(address)
        assert len(recalled) == 128
        assert all(isinstance(val, int) for val in recalled)
        assert all(val in [0, 1] for val in recalled)

    def test_write_invalid_address_size(self):
        """Test that incorrect address size raises ValueError."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.write([0] * 100, [1] * 128)

    def test_write_invalid_memory_size(self):
        """Test that incorrect memory size raises ValueError."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.write([0] * 256, [1] * 100)

    def test_write_non_binary_address(self):
        """Test that non-binary address values raise ValueError."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.write([2] * 256, [1] * 128)

    def test_write_non_binary_memory(self):
        """Test that non-binary memory values raise ValueError."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.write([0] * 256, [3] * 128)

    def test_read_invalid_address_size(self):
        """Test that incorrect address size raises ValueError during read."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.read([0] * 100)

    def test_read_non_binary_address(self):
        """Test that non-binary address values raise ValueError during read."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        
        with pytest.raises(ValueError):
            sdm.read([2] * 256)

    def test_erase_memory(self):
        """Test that erase_memory resets memory count."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103, random_seed=42)
        
        address = [0] * 256
        memory = [1] * 128
        
        sdm.write(address, memory)
        assert sdm.memory_count == 1
        
        sdm.erase_memory()
        assert sdm.memory_count == 0
        
        # Verify locations are preserved by checking dimensions
        assert sdm.num_locations == 1000

    def test_multiple_writes(self):
        """Test multiple write operations."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103, random_seed=42)
        
        for i in range(5):
            address = [i % 2] * 256
            memory = [(i + 1) % 2] * 128
            sdm.write(address, memory)
        
        assert sdm.memory_count == 5

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same behavior."""
        sdm1 = kanerva_sdm.KanervaSDM(256, 128, 1000, 103, random_seed=42)
        sdm2 = kanerva_sdm.KanervaSDM(256, 128, 1000, 103, random_seed=42)
        
        address = [0, 1] * 128
        memory = [1, 0] * 64
        
        sdm1.write(address, memory)
        sdm2.write(address, memory)
        
        result1 = sdm1.read(address)
        result2 = sdm2.read(address)
        
        assert result1 == result2

    def test_repr(self):
        """Test string representation."""
        sdm = kanerva_sdm.KanervaSDM(256, 128, 1000, 103)
        repr_str = repr(sdm)
        
        assert "KanervaSDM" in repr_str
        assert "256" in repr_str
        assert "128" in repr_str
        assert "1000" in repr_str
        assert "103" in repr_str

    def test_version_attribute(self):
        """Test that version attribute exists."""
        assert hasattr(kanerva_sdm, "__version__")
        assert isinstance(kanerva_sdm.__version__, str)
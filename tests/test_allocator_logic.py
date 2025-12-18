
import unittest
from unittest.mock import MagicMock, patch
import threading
from kvcached.page_allocator import PageAllocator

class TestPageAllocatorRemote(unittest.TestCase):
    def setUp(self):
        self.page_size = 2 * 1024 * 1024  # 2MB
        self.num_layers = 16
        # One logical page ID consumes: 2MB * 16 * 2 = 64MB physical
        self.mem_per_layer = 10 * 64 * 1024 * 1024 # 10 logical pages worth
        # So we have 10 virtual pages.
        
        self.shared_config = {0: 128 * 1024 * 1024} # 128MB remote (2 pages)

    @patch('torch.cuda.mem_get_info')
    @patch('kvcached.page_allocator.map_to_kv_tensors')
    @patch('kvcached.page_allocator.unmap_from_kv_tensors')
    @patch('kvcached.page_allocator.MemInfoTracker')
    def test_remote_allocation_accounting(self, mock_tracker, mock_unmap, mock_map, mock_mem_info):
        # Setup: Local memory enough for 5 pages.
        # Logical Page physical size = 64MB.
        # 5 pages = 320MB.
        # Let's say we have 320MB free + headroom.
        # GPU_UTILIZATION is 0.95.
        # Let's simulate exact numbers.
        # _get_local_avail_physical_pages logic:
        # avail_phy = avail - (total * 0.05).
        # We want avail_phy = 320MB.
        # Let total = 1000MB. Headroom = 50MB.
        # Avail = 320 + 50 = 370MB.
        
        mock_mem_info.return_value = (370 * 1024 * 1024, 1000 * 1024 * 1024)
        
        # Init Allocator
        allocator = PageAllocator(
            num_layers=self.num_layers,
            mem_size_per_layer=self.mem_per_layer,
            page_size=self.page_size,
            shared_memory_config=self.shared_config,
            enable_page_prealloc=False # Disable prealloc thread for determinsm
        )
        
        # Verify Init State
        # Local Avail: 5 pages.
        # Remote Avail: 128MB = 2 pages.
        # Total Avail: 7 pages.
        self.assertEqual(allocator._get_local_avail_physical_pages(), 5)
        self.assertEqual(allocator.get_avail_physical_pages(), 7) 
        
        # Alloc 5 pages (Should fill Local)
        pages = []
        for i in range(5):
            p = allocator.alloc_page()
            pages.append(p)
            self.assertEqual(p.phys_dev_id, -1, f"Page {i} should be local")
        
        # Check Avail
        # _get_local IS STATLESS (based on mem_get_info).
        # Since we mocked mem_get_info constant, it thinks it still has 5 pages?
        # WAIT. In real life, allocations consume memory, so mem_get_info drops.
        # I must mock mem_get_info to drop!
        
        current_free = 370 * 1024 * 1024
        page_phy_size = 64 * 1024 * 1024
        
        def update_mem_info():
            nonlocal current_free
            return (current_free, 1000 * 1024 * 1024)
        
        mock_mem_info.side_effect = update_mem_info
        
        # Redo Alloc 5 pages logic with decreasing memory
        for i in range(5):
            # Check before
            expected_local = (current_free - 50*1024*1024) // allocator.page_size // 16 // 2
            # Alloc
            current_free -= page_phy_size # Mimic driver consumption
            
        # Now Local should be 0.
        mock_mem_info.side_effect = lambda: (50 * 1024 * 1024, 1000 * 1024 * 1024) # Just headroom left
        
        self.assertEqual(allocator._get_local_avail_physical_pages(), 0)
        
        # Check Total Avail: Local(0) + Remote(2) = 2.
        self.assertEqual(allocator.get_avail_physical_pages(), 2)
        
        # Allocate 6th page (Should be Remote)
        p6 = allocator.alloc_page()
        self.assertNotEqual(p6.phys_dev_id, -1, "Page 6 should be remote")
        self.assertEqual(p6.phys_dev_id, 0, "Page 6 should be on GPU 0")
        
        # Check Accounting
        # remote_allocated_bytes should be 64MB.
        self.assertEqual(allocator.remote_allocated_bytes, 64 * 1024 * 1024)
        
        # Check Avail: Local(0) + Remote(128-64=64MB=1page) = 1.
        self.assertEqual(allocator.get_avail_physical_pages(), 1)
        
        # Free Remote Page
        allocator.free_page(p6.page_id)
        
        # Check Accounting
        self.assertEqual(allocator.remote_allocated_bytes, 0)
        self.assertEqual(allocator.get_avail_physical_pages(), 2)
        self.assertNotIn(p6.page_id, allocator.page_locations)

        print("Test Passed")

if __name__ == '__main__':
    unittest.main()

# GPU 虚拟内存\n\n参见[英文版](../../en/core-concepts/virtual-memory.md)获取完整技术细节。\n\nkvcached 利用 CUDA 虚拟内存管理（VMM）API 实现 GPU 的操作系统级虚拟内存，以 2MB 页面粒度进行按需物理内存分配。

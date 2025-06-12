import kvcached.controller.client as kvcached_client


def test_charge():
    client = kvcached_client.MemoryInfoClient()
    avail_mem = client.get_available_memory_in_bytes()
    print(f"avail_mem: {avail_mem}")
    print("charge 100")
    assert client.charge(100)
    assert client.get_available_memory_in_bytes() == avail_mem - 100
    print(f"avail_mem after charge: {client.get_available_memory_in_bytes()}")


def test_uncharge():
    client = kvcached_client.MemoryInfoClient()
    avail_mem = client.get_available_memory_in_bytes()
    print(f"avail_mem: {avail_mem}")
    print("charge 100")
    assert client.charge(100)
    print(f"avail_mem after charge: {client.get_available_memory_in_bytes()}")
    print("uncharge 100")
    assert client.uncharge(100)
    assert client.get_available_memory_in_bytes() == avail_mem
    print(
        f"avail_mem after uncharge: {client.get_available_memory_in_bytes()}")


def main():
    test_charge()
    test_uncharge()


if __name__ == "__main__":
    main()

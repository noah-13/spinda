def decode_utf8_bytes_to_str_wrong(bytesring: bytes):
    return "".join([bytes([b]).decode('utf-8') for b in bytesring])

s = "hello!".encode('utf-8')
print(decode_utf8_bytes_to_str_wrong(s))  # This will raise an error
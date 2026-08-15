import unittest

import collect_dataset


class CollectDatasetTests(unittest.TestCase):
    def test_find_esp32_port_prefers_esp32_ports(self):
        ports = [
            "ttyS0",
            "/dev/ttyUSB1",
            "/dev/ttyACM0",
            "/dev/cu.usbserial-123",
        ]
        port = collect_dataset.find_esp32_port(ports)
        self.assertEqual(port, "/dev/ttyACM0")

    def test_parse_sensor_line_accepts_valid_packet(self):
        line = "1,410,500,620,700,800,1,1,0,0,1,0,100,200,300,400,500,600"
        self.assertTrue(collect_dataset.is_valid_sensor_line(line))

    def test_parse_sensor_line_rejects_header_or_invalid_packet(self):
        self.assertFalse(collect_dataset.is_valid_sensor_line("Mode,FlexPinky,..."))
        self.assertFalse(collect_dataset.is_valid_sensor_line("1,2,3"))


if __name__ == "__main__":
    unittest.main()

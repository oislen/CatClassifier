import unittest

from aws.prg_run_ec2_instance import run_ec2_instance
from aws.utilities.EC2Client import EC2Client


class TestRunEC2Instance(unittest.TestCase):
    def test_run_ec2_instance(self):
        # Assuming run_ec2_instance function exists and returns an instance ID
        instance_id = run_ec2_instance()
        self.assertIsNotNone(instance_id)
        self.assertIsInstance(instance_id, str)

    def test_ec2_client_initialization(self):
        # Assuming EC2Client class exists and can be instantiated
        client = EC2Client()
        self.assertIsInstance(client, EC2Client)


if __name__ == "__main__":
    unittest.main()

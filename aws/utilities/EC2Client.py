import json

import boto3
from beartype import beartype


class EC2Client:
    @beartype
    def __init__(self, session_token: str):
        # load aws config
        with open(session_token) as j:
            aws_config = json.loads(j.read())
        # connect to aws boto3
        self.session = boto3.Session(
            aws_access_key_id=aws_config["Credentials"]["AccessKeyId"],
            aws_secret_access_key=aws_config["Credentials"]["SecretAccessKey"],
            aws_session_token=aws_config["Credentials"]["SessionToken"],
            region_name="eu-west-1",
        )
        # generate boto3 s3 connection
        self.client = self.session.client("ec2")

    def create_launch_template(self, launch_template_config) -> dict:
        """Creates an EC2 launch template

        Parameters
        ----------
        launch_template_config : dict
            The launch template to create

        Returns
        -------
        dict
            The EC2 create launch template response
        """
        # create ec2 launch template
        response = self.client.create_launch_template(**launch_template_config)
        return response

    def delete_launch_template(self, launch_template_name) -> dict:
        """Deletes an EC2 launch template

        Parameters
        ----------
        launch_template_name : str
            The launch template name to delete

        Returns
        -------
        dict
            The EC2 delete launch template response
        """
        # delete ec2 launch template
        response = self.client.delete_launch_template(
            launch_template_name=launch_template_name
        )
        return response

    def create_fleet(self, create_fleet_config) -> dict:
        """Creates an EC2 fleet

        Parameters
        ----------
        create_fleet_config : dict
            The EC2 create fleet configs

        Returns
        -------
        dict
            The EC2 create fleet response
        """
        response = self.client.create_fleet(**create_fleet_config)
        return response

    def describe_fleets(self) -> dict:
        """Describes EC2 fleets

        Parameters
        ----------

        Returns
        -------
        dict
            The EC2 describe fleets response
        """
        response = self.client.describe_fleets()
        return response

    def delete_fleets(self, fleet_ids=[], terminate_instances=False) -> dict:
        """Delete an EC2 fleet

        Parameters
        ----------
        fleet_ids : list
            The fleet ids to delete, default is []
        terminate_instances : bool
            Whether to delete the fleets, default is True

        Returns
        -------
        dict
            The EC2 delete fleets EC2 response
        """
        response = self.client.delete_fleets(
            fleet_ids=fleet_ids, terminate_instances=terminate_instances
        )
        return response

    def run_instances(self, run_instances_config) -> dict:
        """Runs an EC2 instance

        Parameters
        ----------
        run_instances_config : dict
            The run instance configurations

        Returns
        -------
        The EC2 run instances response
        """
        response = self.client.run_instances(**run_instances_config)
        return response

    def stop_instances(self, instance_ids=[]) -> dict:
        """Stops EC2 instances

        Parameters
        ----------
        instance_ids : list
            The EC2 instance ids to stop, default is []

        Returns
        -------
        dict
            The EC2 stop instances response
        """
        response = self.client.stop_instances(instance_ids=instance_ids)
        return response

    def terminate_instances(self, instance_ids=[]) -> dict:
        """Terminates EC2 instances

        Parameters
        ----------
        instance_ids : list
            The EC2 instance ids to terminate, default is []

        Returns
        -------
        dict
            The EC2 terminate instances response
        """
        response = self.client.terminate_instances(instance_ids=instance_ids)
        return response

    def describe_instances(self, instance_ids=[], filters=[], max_results=20) -> dict:
        """Describe EC2 instances

        Parameters
        ----------
        instance_ids : list
            The instance ids to describe, default is []
        filters : list
            The filters to apply when describing instances, default is []
        max_results : int
            The maximum number of results to return

        Returns
        -------
        dict
            The EC2 describe instances response
        """
        response = self.client.describe_instances(
            instance_ids=instance_ids, filters=filters, max_results=max_results
        )
        return response

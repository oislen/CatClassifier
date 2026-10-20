import logging

import cons
from beartype import beartype
from utilities.commandline_interface import commandline_interface
from utilities.EC2Client import EC2Client


@beartype
def run_ec2_instance(
    launch: bool = False,
    terminate: bool = False,
    describe: bool = False,
    is_fleet: bool = False,
):
    """

    Parameters
    ----------
    launch : bool
        Whether to launch an ec2 instance using the launch template
    terminate : bool
        Whether to stop and terminate all running ec2 instance
    describe : bool
        Whether to list all recorded instances
    is_fleet : bool
        Whether : the type to launch is a fleet, otherwise single instance

    Returns
    -------
    """
    logging.info("Creating EC2 client.")
    # create EC2 client
    ec2_client = EC2Client(session_token=cons.session_token_fpath)
    # if listing available instances
    if describe:
        if is_fleet:
            logging.info("Listing EC2 fleets")
            response = ec2_client.describe_fleets()
        else:
            logging.info("Listing EC2 instances")
            filters = [
                {"Name": "instance-state-name", "Values": ["running", "pending"]}
            ]
            response = ec2_client.describe_instances(filters=filters)
        logging.info(response)
    # if launch ec2 instance
    if launch:
        logging.info("Launching EC2 instance.")
        try:
            # delete any existing launch template
            ec2_client.delete_launch_template(
                launch_template_name=cons.launch_template_config["launch_template_name"]
            )
        except Exception as e:
            logging.warning(e)
        # create a new launch template
        ec2_client.create_launch_template(cons.launch_template_config)
        if is_fleet:
            # create ec2 fleet
            ec2_client.create_fleet(cons.create_fleet_config)
        else:
            # create ec2 instance
            ec2_client.run_instances(cons.run_instances_config)
        # list any instances
        filters = [{"Name": "instance-state-name", "Values": ["running", "pending"]}]
        response = ec2_client.describe_instances(filters=filters)
        logging.info(response)
    # if terminating ec2 instance
    if terminate:
        if is_fleet:
            logging.info("Terminating EC2 fleets.")
            # list any fleets
            response = ec2_client.describe_fleets()
            # set instance ids to shut down
            fleet_ids = [fleet["FleetId"] for fleet in response["Fleets"]]
            if fleet_ids != []:
                response = ec2_client.delete_fleets(
                    fleet_ids=fleet_ids, terminate_instances=True
                )
            # list any fleets
            response = ec2_client.describe_fleets()
        else:
            logging.info("Terminating EC2 instances.")
            # list any running instances
            filters = [{"Name": "instance-state-name", "Values": ["running"]}]
            response = ec2_client.describe_instances(filters=filters)
            # set instance ids to shut down
            instance_ids = [
                instance["InstanceId"]
                for reservation in response["Reservations"]
                for instance in reservation["Instances"]
            ]
            if instance_ids != []:
                ec2_client.stop_instances(instance_ids=instance_ids)
                ec2_client.terminate_instances(instance_ids=instance_ids)
            # list any running instances
            response = ec2_client.describe_instances(filters=filters)
        logging.info(response)


if __name__ == "__main__":
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)
    # handle input parameters
    input_params_dict = commandline_interface()
    # run main programme
    run_ec2_instance(
        launch=input_params_dict["launch"],
        terminate=input_params_dict["terminate"],
        describe=input_params_dict["describe"],
        is_fleet=input_params_dict["is_fleet"],
    )

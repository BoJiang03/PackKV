import os
from datetime import datetime
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import matplotlib.pyplot as plt


def set_stdout_log():
    # print pwd
    pwd = os.getcwd()
    file_name = sys.argv[0].split("/")[-1].split(".")[0]
    print(f"write log to {pwd}/logs/{file_name}")
    if not os.path.exists(f"{pwd}/logs/{file_name}"):
        os.makedirs(f"{pwd}/logs/{file_name}")
        print(f"create {pwd}/logs/{file_name}")

    # create file {timestamp}.log
    timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    log_file = f"{pwd}/logs/{file_name}/{timestamp}.log"
    print(f"write log to {log_file}")

    # Open file once and use it for both stdout and stderr
    log_handle = open(log_file, "w", buffering=1)  # Line-buffered output
    sys.stdout = log_handle
    sys.stderr = log_handle


def get_logger(file_path):
    # Get file name and prepare paths
    file_name = os.path.basename(file_path).split('.')[0]
    pwd = os.getcwd()
    log_dir = f"{pwd}/logs/{file_name}"

    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    log_file = f"{log_dir}/{timestamp}.log"

    # Create logger with unique name to avoid conflicts
    logger = logging.getLogger(f"{file_name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters and handlers
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(filename)s] %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def block_other_logger(logger):
    # Block other loggers
    for name in logging.root.manager.loggerDict:
        if name != logger.name and name.startswith(logger.name):
            logging.getLogger(name).propagate = False


def register_notify():
    import atexit
    atexit.register(notify_user)


def notify_user():
    # Read email configuration from environment variables
    sender_email = os.getenv("EMAIL_SENDER")
    receiver_email = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PASSWORD")

    # Check if all required environment variables are set
    if not all([sender_email, receiver_email, password]):
        missing_vars = []
        if not sender_email:
            missing_vars.append("EMAIL_SENDER")
        if not receiver_email:
            missing_vars.append("EMAIL_RECEIVER")
        if not password:
            missing_vars.append("EMAIL_PASSWORD")
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        return

    # Create the email subject and body
    subject = "Script Notification"
    command = ' '.join(sys.argv)
    body = f"The script has finished running with the following command:\n{command}"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Connect to the server and send email
    try:
        with smtplib.SMTP("smtp.gmail.com", port=587) as server:
            server.ehlo()
            server.starttls()  # Secure the connection
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Notification email sent successfully")
    except Exception as e:
        print(f"Error: {e}")


def visualize_2d_tensor(tensor, save_path: str = None, dpi: int = 200):
    print(f"Plotting tensor with shape {tensor.shape}")

    # plt.figure(figsize=(20, 20))
    plt.pcolormesh(tensor.cpu(), cmap='viridis', shading='auto')
    plt.colorbar(label='Value')
    plt.title('2D Matrix Visualization')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal')
    # plt.show()
    if save_path:
        print(f"Saving to {save_path}")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()


class JumpOutException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
import smtplib, ssl

# https://realpython.com/python-send-email/


receivers = ['kanojiyamayur@gmail.com']


class SendMail(object):

    def __init__(self):
        self.port = 465  # For SSL

        self.sender = 'www.kmk.com@gmail.com'
        # password = input("Type your password and press enter: ")
        password = 'kmkmkmkmkmayur'

        # Create a secure SSL context
        self.context = ssl.create_default_context()
        self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.port, context=self.context)
        self.server.login(self.sender, password)

    def send_alert(self, body, fall_type):
        try:

            message = """From: From Person <{}>
            To: To Person <{}>
            Subject: Alert | Fall Detected | {}

            {}.
            """.format(self.sender, receivers[0], fall_type, body)
            self.server.sendmail(self.sender, receivers, message)
            print("Successfully sent email")
        except smtplib.SMTPException as err:
            print("Error: unable to send email Reason: {}".format(err))
            exit(0)

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
        try:
            self.server.login(self.sender, password)
        except Exception as err:
            print(
                'Turn Allow less secure apps to ON from following link. Be aware that this makes it easier for others to gain access to your account.')
            print('https://myaccount.google.com/lesssecureapps')

    def send_alert(self, body, fall_type):
        try:

            message = """From: <{}>
            To:  <{}>
            Subject: Alert | Fall Detected | {}

            {}.
            """.format(self.sender, receivers[0], fall_type, body)

            self.server.sendmail(self.sender, receivers, message)
            print("Successfully sent email")
        except smtplib.SMTPException as err:
            print(
                'Turn Allow less secure apps to ON from following link. Be aware that this makes it easier for others to gain access to your account.')
            print('https://myaccount.google.com/lesssecureapps')
            print("Error: unable to send email Reason: {}".format(err))
            exit(0)

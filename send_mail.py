import smtplib
#https://realpython.com/python-send-email/

sender = 'www.kmk.com@gmail.com'
receivers = ['kanojiyamayur@gmail.com']

message = """From: From Person <{}>
To: To Person <{}>
Subject: Fall Detected | Alert

This is a test e-mail message.
""".format(sender, receivers[0])


import smtplib, ssl

port = 465  # For SSL
password = input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(sender, password)
    try:
        server.sendmail(sender, receivers, message)
        print("Successfully sent email")
    except smtplib.SMTPException:
        print("Error: unable to send email")



import pymysql

class UserDetailsDataLayer:

    def __init__(self):
        self.conn = pymysql.connect(host="project-1-users.c8srs7bhoefe.us-east-2.rds.amazonaws.com", port=3306,
                                    user='admin', password="admin_123", db='Login_DB')

    def insert_details(self, firstname, lastname, email, username, password):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO USER_DETAILS (FIRSTNAME, LASTNAME, EMAIL, USERNAME, PASSWORD) VALUES (%s,%s,%s,%s,%s)",
                    (firstname, lastname, email, username, password))
        self.conn.commit()

    def get_details(self):
        cur = self.conn.cursor()
        cur.execute("SELECT *  FROM USER_DETAILS")
        details = cur.fetchall()
        return details

    def get_user_details(self, username):
        cur = self.conn.cursor()
        cur.execute("SELECT *  FROM USER_DETAILS where USERNAME = %s", username)
        details = cur.fetchone()
        return details

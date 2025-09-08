import psycopg2

class Connector():

    def __init__(self, connection_params: dict):
        self.db_name = connection_params.get("name")
        self.host = connection_params.get("host")
        self.user = connection_params.get("user")
        self.pwd = connection_params.get("password")
        self.port = connection_params.get("port", 5432)

    def connect(self):
        try:
            connection = psycopg2.connect(
                database=self.db_name,
                user=self.user,
                password=self.pwd,
                host=self.host,
                port=self.port
            )
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
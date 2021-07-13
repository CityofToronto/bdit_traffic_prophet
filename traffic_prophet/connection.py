"""Base classes for connecting to Postgres database."""

import psycopg2


class Connection:
    # TO DO: this currently has NO FORMAL TEST SUITE, mainly because we'll
    # be replacing it with whatever Kedro cooked up at some point.

    def __init__(self, credentials, tablename):
        self.credentials = credentials
        self.tablename = tablename

    def connect(self):
        return psycopg2.connect(**self.credentials)

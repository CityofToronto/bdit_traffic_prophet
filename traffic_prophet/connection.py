"""Base classes for connecting to Postgres database."""


class Connection:

    def __init__(self):
        raise NotImplementedError('psycopg2 connection mechanism is not '
                                  'yet implemented!')

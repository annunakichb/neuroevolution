import evolution.session as session
from moo.nsga2 import NSGA2

def moo_init():
    session.operationRegistry.register(NSGA2(), NSGA2.name)
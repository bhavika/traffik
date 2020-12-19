import structlog
import datetime


def timestamper(_, __, event_dict):
    event_dict["time"] = datetime.datetime.now().isoformat()
    return event_dict


structlog.configure(processors=[timestamper, structlog.processors.JSONRenderer()])

logger = structlog.get_logger()

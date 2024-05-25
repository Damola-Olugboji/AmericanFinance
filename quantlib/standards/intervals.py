import pytz
from enum import Enum
from dateutil.relativedelta import relativedelta

class Period(Enum):
    """
    Enumeration representing different time periods for supported data granularity.

    Attributes:
        SECOND: Represents a second. Value `s`.
        MINUTE: Represents a minute. Value `m`.
        HOURLY: Represents an hour. Value `h`.
        DAILY: Represents a day. Value `d`.
        WEEKLY: Represents a week. Value `w`.
        MONTHLY: Represents a month. Value `M`.
        YEARLY: Represents a year. Value `y`.
    """
    SECOND = 's'
    MINUTE = 'm'
    HOURLY = 'h'
    DAILY = 'd'
    WEEKLY = 'w'
    MONTHLY = 'M'
    YEARLY = "y"

str_to_period = {member.value: member for member in Period}

def map_granularity_to_relativedelta(granularity, periods):
    """
    Map granularity and number of periods to `dateutil.relativedelta.relativedelta`
    object.

    Args:
        granularity (quantpylib.standards.Period): The granularity of time period.
        periods (int): The number of periods.

    Returns:
        relativedelta: Relative delta representing the time period.

    Raises:
        AssertionError: If the granularity is not a supported type.
    """
    if granularity == Period.SECOND:
        deltatime = relativedelta(seconds= periods)
    if granularity == Period.MINUTE:
        deltatime = relativedelta(minutes= periods)
    if granularity == Period.HOURLY:
        deltatime = relativedelta(hours = periods)
    if granularity == Period.DAILY:
        deltatime = relativedelta(days = periods)
    if granularity == Period.WEEKLY:
        deltatime = relativedelta(days = periods * 7)
    if granularity == Period.MONTHLY:
        deltatime = relativedelta(months = periods)
    if granularity == Period.YEARLY:
        deltatime = relativedelta(years = periods)
    assert(deltatime)
    return deltatime

def get_span(granularity, granularity_multiplier, period_start=None, period_end=None, periods=None):
    """
    Get the time span based on the specified granularity and parameters.
    granularity, granularity_multipler and periods determine the span duration, if provided.

    Args:
        granularity (quantpylib.standards.Period): The granularity of time period.
        granularity_multiplier (int): The multiplier for the granularity.
        period_start (datetime, optional): The start of the time period. Defaults to None.
        period_end (datetime, optional): The end of the time period. Defaults to None.
        periods (int, optional): The number of periods. Defaults to None.

    Returns:
        tuple: A tuple containing the (start,end) of timespan.

    Raises:
        InvalidPeriodConfig: If the period configuration is invalid.
        AssertionError: If the provided parameters are invalid.
    """
    if period_start is not None and period_start.tzinfo is None: period_start = period_start.replace(tzinfo=pytz.utc)
    if period_end is not None and period_end.tzinfo is None: period_end = period_end.replace(tzinfo=pytz.utc)

    assert(not periods or periods > 0)
    assert(not (period_start and period_end) or period_end > period_start)
    deltatime=None
    if periods:
        deltatime = map_granularity_to_relativedelta(granularity, periods * granularity_multiplier)
    if not period_start and not period_end and not periods:
        return None, None
    if not period_start and not period_end and periods:
        raise InvalidPeriodConfig("cannot map period specifications to time window")
    if not period_start and period_end and not periods:
        return None, period_end
    if not period_start and period_end and periods:
        return period_end - deltatime, period_end
    if period_start and not period_end and not periods:
        return period_start, None
    if period_start and not period_end and periods:
        return period_start, period_start + deltatime
    if period_start and period_end and not periods:
        return period_start, period_end
    if period_start and period_end and periods:
        raise InvalidPeriodConfig("cannot map period specifications to time window")

class InvalidPeriodConfig(Exception):
    pass
import asyncio 
import functools
import quantlib.standards.intervals as intervals 

def poller(_func=None, *, tag=None):
    """
    Decorator for instance level methods of `quantpylib.datapoller.base.BasePoller` objects. 
    Let the object be given the variable name `poller_obj`, then decorated instance level methods 
    have augmented default arguments `**kwargs` with following key-pairs:

    Specs:

        src (string): Data source we want to poll for. Defaults to `poller_obj.default_src`.

    Args:
        _func (function, optional): The function to be decorated. Defaults to None.
        tag (str, optional): A tag to identify the poller. Defaults to None.

    Returns:
        The decorated instance level method.
    
    """
    
    poller_args = { 
        "src": None, 
    }
    
    def _wrapper(poller_func):
        @functools.wraps(poller_func)
        async def async_wrapper(poller_obj, *args, **kwargs):
            kwargs = {**poller_args, **kwargs}
            if kwargs['src']: assert kwargs['src'] in poller_obj.pollers
            else: kwargs['src'] = poller_obj.default_src
            try:
                return await poller_func(poller_obj, *args, **kwargs)
            except AttributeError as attr:
                print(attr)
                return None

        @functools.wraps(poller_func)
        def sync_wrapper(poller_obj, *args, **kwargs):
            kwargs = {**poller_args, **kwargs}
            if kwargs['src']: assert kwargs['src'] in poller_obj.pollers
            else: kwargs['src'] = poller_obj.default_src
            try:
                return poller_func(poller_obj, *args, **kwargs)
            except AttributeError as attr:
                print(attr)
                return None
            
        if asyncio.iscoroutinefunction(poller_func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return _wrapper
    else:
        return _wrapper(_func)

def ts_poller(_func=None, *, tag=None, assert_span=True, automap_span=True):
    """
    Decorator for instance level methods of `quantpylib.datapoller.base.BasePoller` objects. 
    Let the object be given the variable name `poller_obj`, then decorated instance level methods 
    have augmented default arguments `**kwargs` with following key-pairs:

    Specs:

        ticker (str): The identifier for the time-series of interest. Defaults to None.
        start (datetime.datetime): The start time of the time-series being polled. Defaults to None.
        end (datetime.datetime): The end time of the time-series being polled. Defaults to None.
        periods (int): Number of periods of granularity_multiplier * granularity of time-series being polled. Defaults to None. 
        granularity (str): Granularity of data being polled. Valid values are ['s','m','h','d','w','M','y']. Defaults to 'd'.
        granularity_multiplier (int): Multiplier for the granularity. For instance, 4 for multiplier and 'h' for granularity implies '4h' candles/periods. Defaults to 1.
        src (string): Data source we want to poll for. Defaults to `poller_obj.default_src`.

    Args:
        _func (function, optional): The function to be decorated. Defaults to None.
        tag (str, optional): A tag to identify the poller. Defaults to None.

    Returns:
        method: The decorated instance level method.

    Args:
        _func (function, optional): The function to be decorated. Defaults to None.
        tag (str, optional): A tag to identify the poller. Defaults to None.
        assert_span (bool, optional): Whether to assert the validity of time span arguments. If `True`, exactly two of `start`, `end`, and `periods` must be specified. Defaults to `True`.
        automap_span (bool, optional): Whether to automatically map (`start`,`end`,`periods`,`granularity`,`granularity_multiplier`) into time span arguments. Defaults to `True`.

    Returns:
        The decorated instance level method.
    """
    ts_poller_args = {
        "ticker": None,
        "start": None, 
        "end": None, 
        "periods": None, 
        "granularity": "d", 
        "granularity_multiplier":1,
        "src": None,
    }
        
    def _wrapper(poller_func):        
        @functools.wraps(poller_func)
        def sync_wrapper(poller_obj, *args, **kwargs):
            """
            Synchronous wrapper for time-series poller functions.

            Args:
                poller_obj: The poller object.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Any: Result of the time-series poller function.
            """
            kwargs = {**ts_poller_args, **kwargs}
            if assert_span and sum(1 for arg in ("start","end","periods") if kwargs[arg] is not None) != 2:
                raise ValueError("Exactly two of 'start':datetime.datetime, 'end':datetime.datetime, and 'periods':int must be specified.")
            kwargs["granularity"] = kwargs["granularity"] if isinstance(kwargs["granularity"], intervals.Period) else intervals.str_to_period[kwargs["granularity"]] 
            if automap_span:
                start,end = intervals.get_span(
                    granularity=kwargs["granularity"],
                    granularity_multiplier=kwargs["granularity_multiplier"],
                    period_start=kwargs["start"],
                    period_end=kwargs["end"],
                    periods=kwargs["periods"]
                )
                kwargs["start"],kwargs["end"] = start,end
            if kwargs['src']: assert kwargs['src'] in poller_obj.pollers
            else: kwargs['src'] = poller_obj.default_src
            return poller_func(poller_obj, *args, **kwargs)
                
        @functools.wraps(poller_func)
        async def async_wrapper(poller_obj, *args, **kwargs):
            """
            Asynchronous wrapper for time-series poller functions.

            Args:
                poller_obj: The poller object.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Any: Result of the time-series poller function.
            """
            kwargs = {**ts_poller_args, **kwargs}
            if assert_span and sum(1 for arg in ("start","end","periods") if kwargs[arg] is not None) != 2:
                raise ValueError("Exactly two of 'start':datetime.datetime, 'end':datetime.datetime, and 'periods':int must be specified.")
            kwargs["granularity"] = kwargs["granularity"] if isinstance(kwargs["granularity"], intervals.Period) else intervals.str_to_period[kwargs["granularity"]] 
            if automap_span:
                start,end = intervals.get_span(
                    granularity=kwargs["granularity"],
                    granularity_multiplier=kwargs["granularity_multiplier"],
                    period_start=kwargs["start"],
                    period_end=kwargs["end"],
                    periods=kwargs["periods"]
                )
                kwargs["start"],kwargs["end"] = start,end
            if kwargs['src']: assert kwargs['src'] in poller_obj.pollers
            else: kwargs['src'] = poller_obj.default_src
            return await poller_func(poller_obj, *args, **kwargs)
                
        if asyncio.iscoroutinefunction(poller_func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return _wrapper
    else:
        return _wrapper(_func)

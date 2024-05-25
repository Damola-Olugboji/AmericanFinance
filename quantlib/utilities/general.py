from datetime import datetime

def standard_to_binance_date(input_date):
    date_obj = datetime.strptime(input_date, '%m/%d/%Y')    
    month_name = date_obj.strftime('%b')
    formatted_date = f"{date_obj.day} {month_name}, {date_obj.year}"
    return formatted_date


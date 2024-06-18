def convert_kelvin(temp_format, temperature_kelvin):
    if temp_format == "celsius":
        return temperature_kelvin - 273.15
    elif temp_format == "fahrenheit":
        return (temperature_kelvin - 273.15) * 9/5 + 32
    return temperature_kelvin

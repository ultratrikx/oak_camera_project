import requests

def fetch_damage_values(url="http://172.20.10.2/"):
    """
    Fetches damage/force values from the specified server URL.
    Returns a list of numbers found in the response text.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        # Extract all numbers (integers or floats) from the text
        import re
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", response.text)
        # Convert to float or int as appropriate
        parsed = [float(n) if '.' in n else int(n) for n in numbers]
        return parsed
    except Exception as e:
        print(f"Error fetching damage values: {e}")
        return []

if __name__ == "__main__":
    values = fetch_damage_values()
    print("Fetched damage/force values:", values)

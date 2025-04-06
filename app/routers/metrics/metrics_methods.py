from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db_ops.db_create import Trains, Trips, USAVPdata, MLData, AnomalyData
from sqlalchemy.orm import selectinload, joinedload
async def fetch_and_process_trip_data(trip_id: int, session: AsyncSession):
    # Fetch trip information including the related train data
    async with session.begin():
        trip_info_query = select(Trips).options(joinedload(Trips.train)).where(Trips.trip_id == trip_id)
        trip_info_result = await session.execute(trip_info_query)
        trip_info = trip_info_result.scalars().first()

        if not trip_info:
            return None, "Trip not found"

        # Now this access should not trigger the MissingGreenlet error
        train_info = {
            "train_id": trip_info.train_id,
            "train_type": trip_info.train.type
        }
        
        # Fetch data from USAVPdata, MLData, and AnomalyData
        usavp_query = select(USAVPdata).where(USAVPdata.trip_id == trip_id)
        ml_data_query = select(MLData).where(MLData.trip_id == trip_id)
        anomaly_data_query = select(AnomalyData).where(AnomalyData.trip_id == trip_id)
        
        # Execute queries
        usavp_data_result = await session.execute(usavp_query)
        ml_data_result = await session.execute(ml_data_query)
        anomaly_data_result = await session.execute(anomaly_data_query)
        
        # Fetch results
        usavp_data = usavp_data_result.scalars().all()
        ml_data = ml_data_result.scalars().all()
        anomaly_data = anomaly_data_result.scalars().all()
        
        # Process the data for plotting, this is simplified and needs to be adjusted
        # based on how you plan to generate the plots
        detections_data = {
            "usavp_data": [serialize_sqlalchemy_obj(obj) for obj in usavp_data],
            "ml_data": [serialize_sqlalchemy_obj(obj) for obj in ml_data],
            "anomaly_data": [serialize_sqlalchemy_obj(obj) for obj in anomaly_data]
        }
        returnable = {
        "train_info": train_info,
        "route": trip_info.route,
        "detections_data": detections_data
    }, None
    return returnable

def serialize_sqlalchemy_obj(sqlalchemy_obj):
    """
    Serialize a SQLAlchemy model instance for JSON response.
    Filters out SQLAlchemy internal attributes and relationships.
    """
    return {key: value for key, value in sqlalchemy_obj.__dict__.items() if not key.startswith('_')}


import plotly.graph_objects as go

def calculate_average_speed(data):
    if len(data) < 2:
        return None  # Need at least two data points to calculate speed

    # Ensure data is sorted by time_track if not already
    data.sort(key=lambda x: x['time_track'])

    # Directly use datetime objects in calculations
    start_time = data[0]['time_track']
    end_time = data[-1]['time_track']
    start_position = data[0]['kilometer_value'] + data[0]['picket_value'] * 0.1
    end_position = data[-1]['kilometer_value'] + data[-1]['picket_value'] * 0.1
    
    # Calculate total time in hours
    total_time_hours = (end_time - start_time).total_seconds() / 3600  # Convert seconds to hours
    
    # Calculate total distance in kilometers
    total_distance_km = end_position - start_position  # In kilometers
    
    # Calculate average speed
    if total_time_hours > 0:
        average_speed = total_distance_km / total_time_hours  # km/h
    else:
        average_speed = 0

    return average_speed


async def analyze_and_visualize_data(usavp_data, ml_data, anomaly_data, request):
    
    # Calculate average speed
    usavp_average_speed = calculate_average_speed(usavp_data)
    ml_average_speed = calculate_average_speed(ml_data)

    # Initialize metrics dictionary with average speed
    metrics = {
        'usavp': {
            'average_speed': usavp_average_speed,
        },
        'ml': {
            'average_speed': ml_average_speed,
        },
        'anomaly_count': len(anomaly_data),
    }
    # Generate Time Series Plot
    time_series_fig = go.Figure()
    for data, name in [(usavp_data, 'USAVP'), (ml_data, 'ML')]:
        if data:
            time_series_fig.add_trace(go.Scatter(
                x=[d['time_track'] for d in data],
                y=[d['kilometer_value'] + d['picket_value'] * 0.1 for d in data],
                mode='lines+markers', name=name))
    time_series_fig.update_layout(title_text="Time Series Analysis")
    time_series_url = generate_plot_html(time_series_fig, 'time_series_plot', request)

    # Calculate the total distance covered for each data point
    usavp_distances = [d['kilometer_value'] + d['picket_value'] * 0.1 for d in usavp_data]
    ml_distances = [d['kilometer_value'] + d['picket_value'] * 0.1 for d in ml_data]

    # Generate Histogram for the calculated distances
    histogram_fig = go.Figure()
    histogram_fig.add_trace(go.Histogram(x=usavp_distances, name='USAVP'))
    histogram_fig.add_trace(go.Histogram(x=ml_distances, name='ML'))
    histogram_fig.update_layout(title_text="Distance Distribution", barmode='overlay')
    histogram_fig.update_traces(opacity=0.75)
    
    # Assuming a function generate_plot_html exists that saves the plot and returns a URL
    histogram_url = generate_plot_html(histogram_fig, 'distance_histogram_plot', request)
      
    return {
        "metrics": metrics,
        "plots": {
            "time_series_url": time_series_url,
            "histogram_url": histogram_url,
        }
    }

def generate_plot_html(fig, filename, request):
    # Placeholder for saving plot as HTML and generating URL
    # Implement saving logic here
    file_path = f"static/{filename}.html"
    fig.write_html(file_path)
    static_base_url = str(request.base_url) + "static"
    plot_url = f"{static_base_url}/{filename}.html"
    return plot_url



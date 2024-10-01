import React, { useEffect, useState } from 'react';
import axios from 'axios'
import logo from './assets/logo.png';
import visual1 from './assets/visual1.png'; // Replace with actual image path
import visual2 from './assets/visual2.png'; // Replace with actual image path
import visual3 from './assets/visual3.png'; // Replace with actual image path
import visual4 from './assets/visual4.png';
import banner from './assets/banner.png';


const Home = () => {
  const [weather, setWeather] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [location, setLocation] = useState({ lat: null, lon: null });
  
  const [state, setState] = useState('');
  const [commodity, setCommodity] = useState('');
  const [duration, setDuration] = useState(null);
  
  const [predictedPrices, setPredictedPrices] = useState([]);
  const [showForecast, setShowForecast] = useState(false);

  const fetchWeather = async (lat, lon) => {
    try {
      const API_KEY = '08a4b3d896ed4756efaf724d8c34945b';
      const weatherRes = await axios.get(
        'https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric'
      );
      const forecastRes = await axios.get(
        'https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric'
      );
      setWeather(weatherRes.data);
      setForecast(forecastRes.data);
    } catch (error) {
      console.error('Error fetching weather data:', error);
    }
  };

  const getUserLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((position) => {
        setLocation({
          lat: 16.43425,
          lon: 81.69845,
        });
        fetchWeather(position.coords.latitude, position.coords.longitude);
      });
    } else {
      console.log('Geolocation is not supported by this browser.');
    }
  };

  useEffect(() => {
    getUserLocation();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        state,
        commodity,
        days: duration,
      });
      console.log('Prediction response:', response.data);
      setPredictedPrices(response.data.predicted_prices);
    } catch (error) {
      console.error('Error fetching prediction data:', error);
    }
  };

  const getFutureDate = (daysFromNow) => {
    const date = new Date();
    const options = { day: '2-digit', month: '2-digit', year: 'numeric' };
    date.setDate(date.getDate() + daysFromNow);
    return date.toLocaleDateString('en-GB', options);
  };

  return (
    <div style={styles.container}>
      <div style={styles.sidebar}>
        <ul style={styles.sidebarList}>
          <img src={logo} alt="Logo" style={styles.logo} />
          <li style={styles.sidebarItem}> <a href='./Home' style={{ textDecoration: 'none', color: 'black' }}>Home</a> </li>
          <li style={styles.sidebarItem}><a href='./Dashboard' style={{ textDecoration: 'none', color: 'black' }}>Dashboard</a></li>
          <li style={styles.sidebarItem}><a href='./Forecast' style={{ textDecoration: 'none', color: 'black' }}>Forecast</a></li>
          <li style={styles.sidebarItem}><a href='./UserProfile' style={{ textDecoration: 'none', color: 'black' }}>User Profile</a></li>
          <li style={styles.sidebarItem}><a href='./AboutUs' style={{ textDecoration: 'none', color: 'black' }}>About Us</a></li> 
          <li style={styles.sidebarItem}><a href='./Login' style={{ textDecoration: 'none', color: 'black' }}><h6>Log out</h6></a></li>
        </ul>
      </div>

      <div style={styles.mainContent}>
        {weather && (
          <div style={styles.weatherSection}>
            <h2>{weather.name} Weather</h2>
            <p>{weather.main.temp}°C, {weather.weather[0].description}</p>
            <button onClick={() => setShowForecast(!showForecast)} style={styles.toggleButton}>
              {showForecast ? 'Hide Forecast' : 'Show Forecast'}
            </button>
            {showForecast && forecast && (
              <div style={styles.forecastGrid}>
                {forecast.list.slice(0, 7).map((item, index) => (
                  <div key={index} style={styles.forecastItem}>
                    <h6><i>{new Date(item.dt_txt).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' })}</i></h6>
                    <p>{item.main.temp}°C</p>
                    <i>{item.weather[0].description}</i>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <header style={styles.header}>
          <center>
            <h1><br></br>Welcome to Stock Optimus Dashboard!</h1>
            <h5>Explore and Forecast prices for desired area,commodity and duration</h5>
          </center>
        </header>
        <div style={styles.mainContent}>
                <h2 style={styles.mainContentHeader}>Dashboard</h2>

                {/* Placeholder divs for visuals */}
                <div style={styles.visualContainer}>
                    <div style={styles.visualBox}>
                        <img src={visual1} alt="Visual 1" style={styles.visualImg} />
                    </div>
                    <div style={styles.visualBox}>
                        <img src={visual2} alt="Visual 2" style={styles.visualImg} />
                    </div>
                    <div style={styles.visualBox}>
                        <img src={visual3} alt="Visual 3" style={styles.visualImg} />
                    </div>
                    <div style={styles.visualBox}>
                        <img src={visual4} alt="Visual 4" style={styles.visualImg} />
                    </div>
                </div>
            </div>

        
      </div>
    </div>
  );
};

export default Home;

// Add your CSS-in-JS styles here
const styles = {
  container: {
    display: 'flex',
    height: '100vh',
  },
  visualContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)', // Two columns layout
    gap: '20px',
},
visualBox: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '20px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
},
visualImg: {
    width: '100%',
    height: 'auto',
    display: 'block',
    borderRadius: '8px',
},

  sidebar: {
    width: '450px',
    backgroundImage: `url('https://naymatcollateral.com/wp-content/uploads/2020/11/banner1-6.jpg')`,
    backgroundRepeat: 'repeat',
    padding: '20px',
  },
  sidebarList: {
    listStyle: 'none',
    padding: 0,
  },
  sidebarItem: {
    marginBottom: '20px',
    cursor: 'pointer',
  },
  logo: {
    width: '100%',
    marginBottom: '20px',
  },
  mainContent: {
    flexGrow: 1,
    padding: '10px',
    overflowY: 'auto',
  },
  weatherSection: {
    marginBottom: '20px',
  },
  toggleButton: {
    marginTop: '10px',
  },
  forecastGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(7, 1fr)',
    gap: '10px',
    marginTop: '10px',
  },
  forecastItem: {
    backgroundColor: '#e0e0e0',
    padding: '10px',
    textAlign: 'center',
  },
  header: {
    marginBottom: '30px',
    backgroundImage: `url(${banner})`,
    backgroundSize: 'cover',  // Ensures the image covers the entire header
    backgroundPosition: 'center',  // Center the image
    color:'black',
    height: '200px',
       
  },
  form: {
    marginBottom: '30px',
  },
  formGroup: {
    marginBottom: '15px',
  },
  input: {
    padding: '10px',
    width: '100%',
    boxSizing: 'border-box',
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    cursor: 'pointer',
  },
  results: {
    marginTop: '20px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  th: {
    borderBottom: '1px solid #ddd',
    padding: '10px',
    textAlign: 'left',
  },
  td: {
    padding: '10px',
    borderBottom: '1px solid #ddd',
  },
};
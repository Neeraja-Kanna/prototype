import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios'
import { useNavigate } from 'react-router-dom';

function Login(){
    const bgImageStyle = {
        backgroundImage: `url('https://naymatcollateral.com/wp-content/uploads/2020/11/banner1-6.jpg')`, // Replace with the correct path to your image
        backgroundSize: 'cover',        // Ensures the image covers the entire screen
        backgroundPosition: 'center',   // Center the image
        backgroundRepeat: 'no-repeat',  // Prevents the image from repeating
        height: '100vh',                // Makes the div as tall as the viewport
        width: '100vw'                  // Makes the div as wide as the viewport  I am writing this to test
    };

    const [email, setEmail] = useState()
    const [password, setPassword] = useState()
    const navigate = useNavigate()

    const handleSubmit = (e) =>{
        e.preventDefault()
        axios.post('http://localhost:3001/login' ,{email,password})
        .then(result=> {console.log(result)
            if(result.data ==="Success"){
                navigate('/home')
            }
           
        })
        .catch(err=>console.log(err))
    }

    return (
        <div style={bgImageStyle} className="d-flex justify-content-center align-items-center bg secondary vh-100">
            <div className="bg-white p-3 rounded w-25">
                <h2 className="text-center mb-4" >Login</h2>
                <form onSubmit={handleSubmit}>
                  
                    <div className="form-group mb-3">
                        <label htmlFor="email"><strong>Email</strong></label>
                        <input
                            type="email"
                            className="form-control rounded-0"
                            autoComplete="off"
                            name="email"
                            placeholder="Enter Email"
                            onChange={(e)=> setEmail(e.target.value)}
                            
                        />
                    </div>
                    <div className="form-group mb-3">
                        <label htmlFor="password"><strong>Password</strong></label>
                        <input
                            type="password"
                            className="form-control rounded-0"
                            name="password"
                            placeholder="Enter Password"
                            onChange={(e)=> setPassword(e.target.value)}
                        />
                    </div>
                    <button type="submit" className="btn btn-success w-100 rounded-0">
                        Login
                    </button>
                    </form>
                <div style={{ color: 'purple' }} className="text-center mt-3">
                    <p>Don't have an account? please sign up</p>
                    <Link to="/register" style={{ color: 'red' }} className="btn btn-default border w-100 bg-light rounded-0 text-decoration-none">
                        Sign Up
                     </Link>
                </div>
            </div>
        </div>
    );
};

export default Login;

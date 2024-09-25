import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios'
import { useNavigate } from 'react-router-dom';

function Signup(){
    const bgImageStyle = {
        backgroundImage: `url('https://naymatcollateral.com/wp-content/uploads/2020/11/banner1-6.jpg')`, // Replace with the correct path to your image
        backgroundSize: 'cover',        // Ensures the image covers the entire screen
        backgroundPosition: 'center',   // Center the image
        backgroundRepeat: 'no-repeat',  // Prevents the image from repeating
        height: '100vh',                // Makes the div as tall as the viewport
        width: '100vw'                  // Makes the div as wide as the viewport
    };

    const [name, setName] = useState()
    const [email, setEmail] = useState()
    const [password, setPassword] = useState()
    const navigate = useNavigate()

    const handleSubmit = (e) =>{
        e.preventDefault()
        axios.post('http://localhost:3001/register' ,{name,email,password})
        .then(result=> {console.log(result)
            navigate('/login')

        })
        .catch(err=>console.log(err))
    }

    return (
        <div style={bgImageStyle} className="d-flex justify-content-center align-items-center bg secondary vh-100">
            <div className="bg-white p-3 rounded w-25">
                <h2 className="text-center mb-4" >Register</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group mb-3">
                        <label htmlFor="name"><strong>Name</strong></label>
                        <input
                            type="text"
                            placeholder="Enter Name"
                            autoComplete="off"
                            name="name"
                            className="form-control rounded-0"
                            onChange={(e)=> setName(e.target.value)}
                           
                        />
                    </div>
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
                        Register
                    </button>
                    </form>
                <div   style={{ color: 'purple' }} className="text-center mt-3">
                    <p>Already Have an Account? please login directly</p>
                    <Link to="/login"  style={{ color: 'Blue'  }} className="btn btn-default border w-100 bg-light rounded-0 text-decoration-none">
                        Login
                     </Link>
                </div>
              
            </div>
        </div>
    );
};

export default Signup;

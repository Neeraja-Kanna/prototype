const mongoose = require('mongoose')

const UsersSchema = new  mongoose.Schema({
    name: String,
    email : String,
    password: String

})

const UsersModel = mongoose.model("registered",UsersSchema )
module.exports = UsersModel
# Mango
Mango is an open-source e-commerce platform built with Django and Python. It is designed to be fast, secure, and scalable, providing an easy-to-use interface for managing products, orders, and customers.

## Features

- Customizable storefront and dashboard
- Support for multiple languages and currencies
- Product categories and attributes
- Product variants and options
- Product reviews and ratings
- User accounts and profiles
- Guest checkout and order tracking
- Payment gateway integrations (Stripe, PayPal, and more)
- Shipping rate calculations (based on weight, price, and destination)
- Tax rate calculations (based on location and product type)
- Coupon codes and discounts
- Email notifications (order confirmation, shipping updates, etc.)
- SEO optimization (URLs, meta tags, sitemaps, etc.)
- Analytics and reporting (Google Analytics, sales reports, etc.)


## Installation
To install Mango, follow these steps:

**1.** Clone the repository:
```
git clone https://github.com/baobabsoluciones/mango.git
Create a virtual environment and activate it:
```


**2.** Create a virtual environment and activate it:
```
python3 -m venv env
source env/bin/activate
Install the requirements:
```


**3.** Install the requirements:
```
pip install -r requirements.txt
Set up the database:
```

**4.** Set up the database:
```
python manage.py migrate
Load sample data (optional):

```

**5.** Load sample data (optional):
```
python manage.py loaddata sample_data.json
Run the development server:
```

**6.** Run the development server:
```
python manage.py runserver
The server will be running at http://127.0.0.1:8000/.
```

## Usage
To use Mango, you can log in to the admin dashboard (http://127.0.0.1:8000/admin/) with the following credentials:

Username: admin
Password: admin
From the dashboard, you can add products, manage orders, view reports, and customize the storefront.

To view the storefront, go to http://127.0.0.1:8000/. You can browse products, add them to the cart, and complete the checkout process.

## Contributing
Contributions to Mango are welcome and appreciated! If you would like to contribute, please follow these steps:

Fork the repository:
`git clone https://github.com/your-username/mango.git`

Create a new branch:
`git checkout -b my-feature-branch`

Make changes and commit them:
```
git add .
git commit -m "Add my feature"
```


Push your changes to your fork:
```
git push origin my-feature-branch
Create a pull request from your fork to the main repository.
```

License
Mango is licensed under the MIT License. See LICENSE for more information.

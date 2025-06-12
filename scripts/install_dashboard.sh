#!/bin/bash

# YinshML Dashboard Installation Script
# This script sets up the dashboard as a systemd service for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/yinshml"
SERVICE_USER="yinsh"
SERVICE_GROUP="yinsh"
SERVICE_NAME="yinsh-dashboard"
PYTHON_VERSION="3.10"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check for required commands
    local deps=("python3" "pip3" "systemctl" "useradd" "git")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency '$dep' not found"
            exit 1
        fi
    done
    
    # Check Python version
    local python_ver=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$python_ver" < "3.8" ]]; then
        log_error "Python 3.8+ required, found $python_ver"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

create_user() {
    log_info "Creating service user and group..."
    
    if ! getent group "$SERVICE_GROUP" > /dev/null 2>&1; then
        groupadd --system "$SERVICE_GROUP"
        log_success "Created group: $SERVICE_GROUP"
    else
        log_info "Group $SERVICE_GROUP already exists"
    fi
    
    if ! getent passwd "$SERVICE_USER" > /dev/null 2>&1; then
        useradd --system --gid "$SERVICE_GROUP" --home-dir "$INSTALL_DIR" \
                --no-create-home --shell /bin/false "$SERVICE_USER"
        log_success "Created user: $SERVICE_USER"
    else
        log_info "User $SERVICE_USER already exists"
    fi
}

setup_directories() {
    log_info "Setting up installation directories..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy project files
    if [[ -f "dashboard/app.py" ]]; then
        log_info "Copying project files to $INSTALL_DIR..."
        cp -r . "$INSTALL_DIR/"
        
        # Set ownership
        chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
        
        # Set permissions
        chmod -R 755 "$INSTALL_DIR"
        chmod -R 644 "$INSTALL_DIR"/*.py "$INSTALL_DIR"/**/*.py 2>/dev/null || true
        
        log_success "Project files copied and permissions set"
    else
        log_error "dashboard/app.py not found. Run this script from the project root."
        exit 1
    fi
}

setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    sudo -u "$SERVICE_USER" python3 -m venv venv
    
    # Upgrade pip
    sudo -u "$SERVICE_USER" ./venv/bin/pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        sudo -u "$SERVICE_USER" ./venv/bin/pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found, skipping dependency installation"
    fi
}

generate_secrets() {
    log_info "Generating secure secrets..."
    
    # Generate random secrets
    SECRET_KEY=$(openssl rand -hex 32)
    JWT_SECRET_KEY=$(openssl rand -hex 32)
    ADMIN_PASSWORD=$(openssl rand -base64 12)
    
    # Create environment file
    cat > "$INSTALL_DIR/.env" << EOF
# YinshML Dashboard Environment Configuration
SECRET_KEY=$SECRET_KEY
JWT_SECRET_KEY=$JWT_SECRET_KEY
DEFAULT_ADMIN_PASSWORD=$ADMIN_PASSWORD
FLASK_ENV=production
FLASK_DEBUG=false
LOG_LEVEL=info
EOF
    
    chown "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    
    log_success "Environment configuration created"
    log_warning "Default admin password: $ADMIN_PASSWORD"
    log_warning "Please save this password and change it after first login!"
}

install_systemd_service() {
    log_info "Installing systemd service..."
    
    # Update service file with actual paths
    sed -e "s|/opt/yinshml|$INSTALL_DIR|g" \
        -e "s|User=yinsh|User=$SERVICE_USER|g" \
        -e "s|Group=yinsh|Group=$SERVICE_GROUP|g" \
        scripts/yinsh-dashboard.service > /etc/systemd/system/$SERVICE_NAME.service
    
    # Set service file permissions
    chmod 644 /etc/systemd/system/$SERVICE_NAME.service
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable $SERVICE_NAME
    
    log_success "Systemd service installed and enabled"
}

setup_firewall() {
    log_info "Configuring firewall (if ufw is available)..."
    
    if command -v ufw &> /dev/null; then
        # Allow dashboard port
        ufw allow 5000/tcp comment "YinshML Dashboard"
        log_success "Firewall rule added for port 5000"
    else
        log_warning "ufw not found, please manually configure firewall to allow port 5000"
    fi
}

create_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/yinsh-dashboard << EOF
/var/log/yinsh-dashboard.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_GROUP
    postrotate
        systemctl reload $SERVICE_NAME
    endscript
}
EOF
    
    log_success "Log rotation configured"
}

start_service() {
    log_info "Starting YinshML Dashboard service..."
    
    systemctl start $SERVICE_NAME
    
    # Wait a moment for service to start
    sleep 3
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        log_success "Dashboard service started successfully"
        log_info "Service status:"
        systemctl status $SERVICE_NAME --no-pager -l
    else
        log_error "Failed to start dashboard service"
        log_info "Check logs with: journalctl -u $SERVICE_NAME -f"
        exit 1
    fi
}

print_summary() {
    echo
    echo "=============================================="
    echo -e "${GREEN}YinshML Dashboard Installation Complete!${NC}"
    echo "=============================================="
    echo
    echo "Service Information:"
    echo "  - Service Name: $SERVICE_NAME"
    echo "  - Install Directory: $INSTALL_DIR"
    echo "  - Service User: $SERVICE_USER"
    echo "  - Dashboard URL: http://localhost:5000"
    echo
    echo "Default Admin Credentials:"
    echo "  - Username: admin"
    echo "  - Password: $ADMIN_PASSWORD"
    echo
    echo "Service Management Commands:"
    echo "  - Start:   sudo systemctl start $SERVICE_NAME"
    echo "  - Stop:    sudo systemctl stop $SERVICE_NAME"
    echo "  - Restart: sudo systemctl restart $SERVICE_NAME"
    echo "  - Status:  sudo systemctl status $SERVICE_NAME"
    echo "  - Logs:    sudo journalctl -u $SERVICE_NAME -f"
    echo
    echo "Configuration:"
    echo "  - Environment: $INSTALL_DIR/.env"
    echo "  - Service File: /etc/systemd/system/$SERVICE_NAME.service"
    echo
    echo -e "${YELLOW}Important:${NC} Please change the default admin password after first login!"
    echo
}

# Main installation process
main() {
    echo "=============================================="
    echo "YinshML Dashboard Installation Script"
    echo "=============================================="
    echo
    
    check_root
    check_dependencies
    create_user
    setup_directories
    setup_python_environment
    generate_secrets
    install_systemd_service
    setup_firewall
    create_log_rotation
    start_service
    print_summary
}

# Handle script arguments
case "${1:-install}" in
    "install")
        main
        ;;
    "uninstall")
        log_info "Uninstalling YinshML Dashboard..."
        systemctl stop $SERVICE_NAME 2>/dev/null || true
        systemctl disable $SERVICE_NAME 2>/dev/null || true
        rm -f /etc/systemd/system/$SERVICE_NAME.service
        systemctl daemon-reload
        rm -rf "$INSTALL_DIR"
        userdel "$SERVICE_USER" 2>/dev/null || true
        groupdel "$SERVICE_GROUP" 2>/dev/null || true
        rm -f /etc/logrotate.d/yinsh-dashboard
        log_success "YinshML Dashboard uninstalled"
        ;;
    "status")
        systemctl status $SERVICE_NAME
        ;;
    "logs")
        journalctl -u $SERVICE_NAME -f
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status|logs}"
        echo "  install   - Install and start the dashboard service"
        echo "  uninstall - Remove the dashboard service and files"
        echo "  status    - Show service status"
        echo "  logs      - Show service logs"
        exit 1
        ;;
esac 
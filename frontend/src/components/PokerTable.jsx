import React, { useState, useEffect } from 'react'
import './PokerTable.css'

const API_BASE_URL = 'http://localhost:5001/api'

const PokerTable = () => {
  const [gameState, setGameState] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [chatMessages, setChatMessages] = useState([
    { type: 'system', text: 'Welcome to Alpha Poker Zero!', timestamp: new Date() },
    { type: 'system', text: 'Ask me for help with your current position. Try: "What should I do?" or "Should I call?"', timestamp: new Date() }
  ])
  const [chatInput, setChatInput] = useState('')
  const chatMessagesEndRef = React.useRef(null)
  const [showRaiseSelector, setShowRaiseSelector] = useState(false)
  const [raiseAmount, setRaiseAmount] = useState(0)
  const [lastBotAction, setLastBotAction] = useState(null)
  const [botActionLog, setBotActionLog] = useState([])

  const scrollToBottom = () => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  React.useEffect(() => {
    scrollToBottom()
  }, [chatMessages])

  const getCurrentPositionInfo = () => {
    if (!gameState) return null
    
    const currentPlayer = gameState.players.find(p => p.isCurrentPlayer)
    if (!currentPlayer) return null

    const suitSymbols = {
      'hearts': '♥',
      'diamonds': '♦',
      'clubs': '♣',
      'spades': '♠'
    }

    const positionInfo = {
      myCards: currentPlayer.cards.map(c => 
        c.suit === 'back' ? '??' : `${c.rank}${suitSymbols[c.suit] || ''}`
      ).join(' '),
      myChips: currentPlayer.chips,
      pot: gameState.pot,
      currentBet: gameState.currentBet,
      gameStage: gameState.gameStage,
      communityCards: gameState.communityCards.length > 0 
        ? gameState.communityCards.map(c => `${c.rank}${suitSymbols[c.suit] || ''}`).join(' ')
        : 'None',
      opponentChips: gameState.players.find(p => !p.isCurrentPlayer)?.chips || 0,
      blinds: `${gameState.smallBlind}/${gameState.bigBlind}`
    }

    return `Current Position:
- My Cards: ${positionInfo.myCards}
- My Chips: ${positionInfo.myChips}
- Pot: ${positionInfo.pot}
- Current Bet: ${positionInfo.currentBet}
- Stage: ${positionInfo.gameStage}
- Community Cards: ${positionInfo.communityCards}
- Opponent Chips: ${positionInfo.opponentChips}
- Blinds: ${positionInfo.blinds}`
  }

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return

    const userMessage = {
      type: 'user',
      text: chatInput,
      timestamp: new Date()
    }

    setChatMessages(prev => [...prev, userMessage])
    setChatInput('')

    // Get AI advice from backend
    try {
      const response = await fetch(`${API_BASE_URL}/game/advice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const data = await response.json()

      if (data.success) {
        const adviceMessage = {
          type: 'assistant',
          text: data.advice || 'No advice available at this time.',
          timestamp: new Date()
        }
        setChatMessages(prev => [...prev, adviceMessage])
      } else {
        const errorMessage = {
          type: 'assistant',
          text: `Error: ${data.error || 'Failed to get advice'}`,
          timestamp: new Date()
        }
        setChatMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Error getting advice:', error)
      const errorMessage = {
        type: 'assistant',
        text: 'Sorry, I encountered an error getting advice. Please try again.',
        timestamp: new Date()
      }
      setChatMessages(prev => [...prev, errorMessage])
    }
  }

  const handleQuickHelp = async () => {
    const helpText = 'What should I do in this position?'
    const userMessage = {
      type: 'user',
      text: helpText,
      timestamp: new Date()
    }
    
    setChatMessages(prev => [...prev, userMessage])
    
    // Get AI advice from backend
    try {
      const response = await fetch(`${API_BASE_URL}/game/advice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const data = await response.json()

      if (data.success) {
        const adviceMessage = {
          type: 'assistant',
          text: data.advice || 'No advice available at this time.',
          timestamp: new Date()
        }
        setChatMessages(prev => [...prev, adviceMessage])
      } else {
        const errorMessage = {
          type: 'assistant',
          text: `Error: ${data.error || 'Failed to get advice'}`,
          timestamp: new Date()
        }
        setChatMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Error getting advice:', error)
      const errorMessage = {
        type: 'assistant',
        text: 'Sorry, I encountered an error getting advice. Please try again.',
        timestamp: new Date()
      }
      setChatMessages(prev => [...prev, errorMessage])
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage()
    }
  }

  // Initialize game on mount
  useEffect(() => {
    startNewGame()
  }, [])

  const startNewGame = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/game/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      const data = await response.json()
      if (data.success) {
        setGameState(data.state)
      } else {
        setError(data.error || 'Failed to start game')
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure backend is running on port 5001.')
      console.error('Error starting game:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchGameState = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/game/state`)
      const data = await response.json()
      if (data.success) {
        setGameState(data.state)
      }
    } catch (err) {
      console.error('Error fetching game state:', err)
    }
  }

  const handleAction = async (action, amount = null) => {
    if (loading || !gameState) return
    
    // If raise, show selector first
    if (action === 'raise' && !showRaiseSelector) {
      setShowRaiseSelector(true)
      const currentPlayer = gameState.players.find(p => p.isCurrentPlayer)
      const toCall = gameState.currentBet - (currentPlayer?.currentBet || 0)
      // Min raise is 2x the current bet (or call if no bet)
      const minRaise = Math.max(gameState.currentBet * 2, toCall + gameState.bigBlind)
      const maxRaise = (currentPlayer?.chips || 0) + toCall
      setRaiseAmount(Math.min(minRaise, maxRaise))
      return
    }
    
    setLoading(true)
    setError(null)
    setShowRaiseSelector(false)
    
    try {
      const response = await fetch(`${API_BASE_URL}/game/action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          action,
          amount: amount || (action === 'raise' ? raiseAmount : null)
        })
      })
      
      const data = await response.json()
      
      if (data.success) {
        setGameState(data.state)
        setShowRaiseSelector(false)
        
        // Store bot action info if available and add to log
        if (data.state.lastAction && data.state.players.find(p => p.id === 2 && p.isCurrentPlayer === false)) {
          const botPlayer = data.state.players.find(p => p.id === 2)
          let actionText = ''
          
          if (data.botRaiseAmount !== undefined && data.botRaiseAmount !== null) {
            actionText = `Bot raised by ${data.botRaiseAmount}`
          } else if (data.state.lastAction === 'raise') {
            const raiseAmount = data.state.currentBet - (botPlayer?.currentBet || 0)
            actionText = `Bot raised by ${raiseAmount}`
          } else if (data.state.lastAction === 'call') {
            const callAmount = data.state.currentBet - (botPlayer?.currentBet || 0)
            actionText = callAmount > 0 ? `Bot called ${callAmount}` : 'Bot checked'
          } else if (data.state.lastAction === 'check') {
            actionText = 'Bot checked'
          } else if (data.state.lastAction === 'fold') {
            actionText = 'Bot folded'
          }
          
          if (actionText) {
            // Add to log
            const logEntry = {
              action: actionText,
              timestamp: new Date(),
              stage: data.state.gameStage,
              pot: data.state.pot
            }
            setBotActionLog(prev => [...prev, logEntry].slice(-20)) // Keep last 20 actions
            setLastBotAction(actionText)
            
            // Clear bot action message after 3 seconds
            setTimeout(() => {
              setLastBotAction(null)
            }, 3000)
          }
        }
        
        // If hand is over, automatically start new hand immediately
        if (data.handOver) {
          setTimeout(() => {
            setLastBotAction(null)
            setBotActionLog([]) // Clear log for new hand
            startNewGame()
          }, 2000) // Reduced to 2 seconds
        }
      } else {
        // If error is "Hand is over" or "No game in progress", automatically start a new hand
        if (data.error && (
          data.error.includes('Hand is over') || 
          data.error.includes('hand is over') ||
          data.error.includes('No game in progress')
        )) {
          setTimeout(() => {
            startNewGame()
          }, 1000)
        } else {
          setError(data.error || 'Failed to make action')
          // Try to recover by fetching current state
          setTimeout(() => {
            fetchGameState()
          }, 1000)
        }
      }
    } catch (err) {
      setError('Failed to connect to server')
      console.error('Error making action:', err)
    } finally {
      setLoading(false)
    }
  }

  // Show loading or error state
  if (!gameState && loading) {
    return (
      <div className="poker-room" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <div style={{ color: '#fff', fontSize: '24px' }}>Loading game...</div>
      </div>
    )
  }

  // Auto-retry on "Hand is over" errors
  if (error && (error.includes('Hand is over') || error.includes('hand is over'))) {
    useEffect(() => {
      const timer = setTimeout(() => {
        setError(null)
        startNewGame()
      }, 1000)
      return () => clearTimeout(timer)
    }, [error])
    return null // Don't show error screen, just auto-restart
  }

  if (error) {
    return (
      <div className="poker-room" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '20px' }}>
        <div style={{ color: '#ff6b6b', fontSize: '20px' }}>{error}</div>
        <button 
          onClick={startNewGame}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
        >
          Retry
        </button>
      </div>
    )
  }

  if (!gameState) {
    return (
      <div className="poker-room" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <button 
          onClick={startNewGame}
          style={{ padding: '15px 30px', fontSize: '18px', cursor: 'pointer' }}
        >
          Start New Game
        </button>
      </div>
    )
  }

  // Calculate call amount
  const currentPlayer = gameState.players.find(p => p.isCurrentPlayer)
  const callAmount = currentPlayer ? gameState.currentBet - (currentPlayer.currentBet || 0) : 0

  return (
    <div className="poker-room">
      {/* Top Header */}
      <div className="poker-header">
        <div className="header-left">
          <div className="logo">ALPHA POKER ZERO</div>
          <button className="btn-guest">Guest</button>
          <button className="btn-signin">Sign In</button>
        </div>
        <div className="header-right">
          <div className="game-info">
            <div className="owner-info">OWNER: JACK</div>
            <div className="game-type">NLH ~ 10 / 20</div>
          </div>
          <div className="header-controls">
            <button className="icon-btn">🔊</button>
            <button className="icon-btn">⏸</button>
            <button className="icon-btn">⏹</button>
          </div>
        </div>
      </div>

      {/* Main Table Area */}
      <div className="table-container">
        <div className="poker-table">

        <div className="table-text">
            <div className="table-title">Alpha Poker Zero</div>
            <div className="table-subtitle">AI-Powered Poker</div>
          </div>

          {/* Winner Announcement */}
          {gameState.isTerminal && gameState.winnerName && (
            <div className="winner-announcement">
              <div className="winner-text">{gameState.winnerName.toUpperCase()} WINS!</div>
              <div className="winner-pot">Pot: {gameState.pot}</div>
            </div>
          )}




          {/* Community Cards Area */}
          <div className="community-cards">
            {gameState.communityCards.map((card, idx) => (
              <div key={idx} className="community-card">
                {card.rank && card.suit ? (
                  <>
                    <div className={`card-rank-top ${card.suit}`}>{card.rank}</div>
                    <div className={`card-suit-center ${card.suit}`}>
                      {card.suit === 'hearts' ? '♥' : card.suit === 'diamonds' ? '♦' : card.suit === 'clubs' ? '♣' : '♠'}
                    </div>
                    <div className={`card-rank-bottom ${card.suit}`}>{card.rank}</div>
                  </>
                ) : (
                  <div className="card-back">ALPHA</div>
                )}
              </div>
            ))}
          </div>

          {/* Players */}
          {gameState.players.map((player) => (
            <div key={player.id} className={`player-seat player-${player.position}`}>
              <div className={`player-cards ${player.isCurrentPlayer ? 'current-player' : ''}`}>
                {player.cards.map((card, idx) => (
                  <div key={idx} className="player-card">
                    {card.suit === 'back' ? (
                      <div className="card-back-face">
                        <div className="card-back-text">ALPHA</div>
                      </div>
                    ) : (
                      <>
                        <div className={`card-rank ${card.suit}`}>{card.rank}</div>
                        <div className={`card-suit ${card.suit}`}>
                          {card.suit === 'hearts' ? '♥' : card.suit === 'diamonds' ? '♦' : card.suit === 'clubs' ? '♣' : '♠'}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
              <div className="player-info">
                <div className="player-name">
                  {player.name}
                  {player.isCurrentPlayer && (
                    <>
                      <span className="player-emoji">😍</span>
                      <span className="player-emoji">😊</span>
                    </>
                  )}
                </div>
                <div className="player-chips">{player.chips}</div>
              </div>
            </div>
          ))}

          {/* Footer Text */}
          <div className="table-footer">
            <div className="footer-brand">ALPHA POKER ZERO</div>
            {/* Pot Display at Bottom */}
            <div className="pot-display-bottom">
              <div className="pot-amount-bottom">{gameState.pot}</div>
              <div className="pot-label-bottom">total {gameState.pot}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Left Sidebar */}
      <div className="sidebar-left">
        <button className="menu-btn">☰</button>
        <div className="menu-options">OPTIONS</div>
        <button className="menu-btn">←</button>
        <button className="menu-btn">👤</button>
      </div>

      {/* Action Panel (Bottom Right) */}
      {gameState.players.find(p => p.isCurrentPlayer) && (
        <div className="action-panel">
          <div className="turn-indicator">
            <div className="turn-badge">YOUR TURN</div>
            <div className="extra-time">EXTRA TIME ACTIVATED</div>
          </div>
          {showRaiseSelector ? (
            <div className="raise-selector">
              <div className="raise-selector-header">Select Raise Amount</div>
              <div className="raise-amount-display">{raiseAmount}</div>
              <input
                type="range"
                className="raise-slider"
                min={gameState.currentBet * 2 - (currentPlayer?.currentBet || 0)}
                max={currentPlayer?.chips || 0}
                value={raiseAmount}
                onChange={(e) => setRaiseAmount(parseInt(e.target.value))}
              />
              <div className="raise-buttons">
                <button
                  className="action-btn raise-confirm-btn"
                  onClick={() => handleAction('raise', raiseAmount)}
                  disabled={loading}
                >
                  RAISE {raiseAmount}
                </button>
                <button
                  className="action-btn raise-cancel-btn"
                  onClick={() => setShowRaiseSelector(false)}
                >
                  CANCEL
                </button>
              </div>
            </div>
          ) : (
            <div className="action-buttons">
              <button 
                className="action-btn call-btn"
                onClick={() => handleAction('call')}
                disabled={loading}
              >
                {callAmount > 0 ? `CALL ${callAmount}` : 'CHECK'}
              </button>
              <button 
                className="action-btn raise-btn"
                onClick={() => handleAction('raise')}
                disabled={loading}
              >
                RAISE
              </button>
              <button 
                className="action-btn fold-btn"
                onClick={() => handleAction('fold')}
                disabled={loading}
              >
                FOLD
              </button>
            </div>
          )}
          {lastBotAction && (
            <div className="bot-action-message">{lastBotAction}</div>
          )}
        </div>
      )}

      {/* Bot Action Log Panel */}
      <div className="bot-action-log-panel">
        <div className="log-header">BOT ACTION LOG</div>
        <div className="log-entries">
          {botActionLog.length === 0 ? (
            <div className="log-empty">No bot actions yet</div>
          ) : (
            botActionLog.map((entry, idx) => (
              <div key={idx} className="log-entry">
                <div className="log-action">{entry.action}</div>
                <div className="log-meta">
                  {entry.stage} • Pot: {entry.pot} • {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Chat/Help Panel (Bottom Left) */}
      <div className="chat-panel">
        <div className="chat-header">
          <span>POKER HELPER</span>
          <button className="quick-help-btn" onClick={handleQuickHelp}>Get Help</button>
        </div>
        <div className="chat-messages" id="chat-messages">
          {chatMessages.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.type}`}>
              <div className="message-content">{msg.text}</div>
              <div className="message-time">
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))}
          <div ref={chatMessagesEndRef} />
        </div>
        <div className="chat-input-area">
          <input 
            type="text" 
            className="chat-input" 
            placeholder="Ask for help with your position..." 
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button className="chat-send-btn" onClick={handleSendMessage}>
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

export default PokerTable


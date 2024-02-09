// external crates
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Complete collection of MCNP particle variants
///
/// The particle is set to [Particle::Unknown] by default, and can be inferred
/// from several possible identifiers. [Particle] implements `TryFrom<&str>`
/// and `TryFrom<u8>` to ensure the failing case is handled.
///
/// ```rust
/// # use meshtal::mesh::Particle;
/// // From the particle symbol
/// assert_eq!(Particle::Alpha, Particle::try_from("a").unwrap());
///
/// // From the particle number
/// assert_eq!(Particle::Alpha, Particle::try_from("34").unwrap());
/// assert_eq!(Particle::Alpha, Particle::try_from(34).unwrap());
///
/// // From the meshtal tag
/// assert_eq!(Particle::Alpha, Particle::try_from("alpha").unwrap());
///
/// // From the full name given in the user manual
/// assert_eq!(Particle::Alpha, Particle::try_from("alpha particle").unwrap());
/// ```
///
/// If the desired behaviour is simply to set any failed conversions to
/// [Particle::Unknown], then the `from_str()` and `from_id()` associated
/// functions are implemented for convenience.
///
/// ```rust
/// # use meshtal::mesh::Particle;
/// // Failing convenience functions are  set to the Unknown variant
/// assert_eq!(Particle::Unknown, Particle::from_str("invalid string"));
/// assert_eq!(Particle::Unknown, Particle::from_id(56));
/// ```
///
/// For reference, a full list of valid MCNP particle identifiers is shown below:
///
/// | ID | Name                                  | Symbol   | Mesh Tag  |
/// | -- | ------------------------------------- | -------- | --------- |
/// | 0  | unknown (special meshtal case)        | NONE     | unknown   |
/// | 1  | neutron                               | n        | neutron   |
/// | 2  | photon                                | p        | photon    |
/// | 3  | electron                              | e        | electron  |
/// | 4  | negative muon                         | \|       | mu_minus  |
/// | 5  | anti neutron                          | q        | Aneutron  |
/// | 6  | electron neutrino                     | u        | nu_e      |
/// | 7  | muon neutrino                         | v        | nu_m      |
/// | 8  | positron                              | f        | *NONE     |
/// | 9  | proton                                | h        | proton    |
/// | 10 | lambda baryon                         | l        | lambda0   |
/// | 11 | positive sigma baryon                 | +        | sigma+    |
/// | 12 | negative sigma baryon                 | -        | sigma-    |
/// | 13 | cascade; xi baryon                    | x        | xi0       |
/// | 14 | negative cascade; negative xi baryon  | y        | xi_minus  |
/// | 15 | omega baryon                          | o        | omega-    |
/// | 16 | positive muon                         | !        | mu_plus   |
/// | 17 | anti electron neutrino                | <        | Anu_e     |
/// | 18 | anti muon neutrino                    | >        | Anu_m     |
/// | 19 | anti proton                           | g        | Aproton   |
/// | 20 | positive pion                         | /        | pi_plus   |
/// | 21 | neutral pion                          | z        | pi_zero   |
/// | 22 | positive kaon                         | k        | k_plus    |
/// | 23 | kaon, short                           | %        | k0_short  |
/// | 24 | kaon, long                            | ^        | k0_long   |
/// | 25 | anti lambda baryon                    | b        | Alambda0  |
/// | 26 | anti positive sigma baryon            | _        | Asigma+   |
/// | 27 | anti negative sigma baryon            | ~        | Asigma-   |
/// | 28 | anti cascade; anti neutral xi baryon  | c        | Axi0      |
/// | 29 | positive cascade; positive xi baryon  | w        | xi_plus   |
/// | 30 | anti omega                            | @        | Aomega-   |
/// | 31 | deuteron                              | d        | deuteron  |
/// | 32 | triton                                | t        | triton    |
/// | 33 | helion                                | s        | helion    |
/// | 34 | alpha particle                        | a        | alpha     |
/// | 35 | negative pion                         | *        | pi_minus  |
/// | 36 | negative kaon                         | ?        | k_minus   |
/// | 37 | heavy ions                            | #        | heavyion  |
///
/// *Note that the positron particle designator is invalid on the FMESH card
/// because it is treated as an electron. It therefore has no meshtal output
/// tag.  
#[repr(u8)]
#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Deserialize, Serialize)]
pub enum Particle {
    #[default]
    Unknown = 0,
    Neutron = 1,
    Photon = 2,
    Electron = 3,
    NegativeMuon = 4,
    AntiNeutron = 5,
    ElectronNeutrino = 6,
    MuonNeutrino = 7,
    Positron = 8,
    Proton = 9,
    LambdaBaryon = 10,
    PosSigmaBaryon = 11,
    NegSigmaBaryon = 12,
    XiBaryon = 13,
    NegXiBaryon = 14,
    OmegaBaryon = 15,
    PosMuon = 16,
    AntiElectronNeutrino = 17,
    AntiMuonNeutrino = 18,
    AntiProton = 19,
    PosPion = 20,
    NeuPion = 21,
    PosKaon = 22,
    ShortKaon = 23,
    LongKaon = 24,
    AntiLambdaBaryon = 25,
    AntiPosSigmaBaryon = 26,
    AntiNegSigmaBaryon = 27,
    AntiNeuXiBaryon = 28,
    PosXiBaryon = 29,
    AntiOmega = 30,
    Deuteron = 31,
    Triton = 32,
    Helion = 33,
    Alpha = 34,
    NegPion = 35,
    NegKaon = 36,
    HeavyIon = 37,
}

impl Particle {
    /// An alternative to using `Neutron as u8`
    ///
    /// For the case where `particle.id()` is prefereable to `particle as u8`,
    /// which may be more intuitive or imporve code readability. Panics on
    /// invalid values.
    ///
    /// ```rust
    /// # use meshtal::mesh::Particle;
    /// // These two statements are equivalent
    /// assert_eq!(Particle::Electron as u8, Particle::Electron.id());
    /// ```
    #[inline]
    pub fn id(&self) -> u8 {
        *self as u8
    }

    /// Convert from any valid particle id
    ///
    /// If the value given is outside of the number of possible variant, the
    /// returned value will be [Particle::Unknown].
    ///
    /// ```rust
    /// # use meshtal::mesh::Particle;
    /// // From a valid particle id
    /// assert_eq!(Particle::Neutron, Particle::from_id(1));
    ///
    /// // Invalid inputs return the Unknown variant (0-37 are valid)
    /// assert_eq!(Particle::Unknown, Particle::from_id(41));
    /// ```
    pub fn from_id(s: u8) -> Self {
        Self::try_from(s).unwrap_or(Self::Unknown)
    }

    /// Convert from any valid designator, name, or meshtal output tag
    ///
    /// If the particle type can not be inferred from the provided string the
    /// [Particle::Unknown] variant is returned. All inputs are insensitive to
    /// case.
    ///
    /// ```rust
    /// # use meshtal::mesh::Particle;
    /// // From the particle symbol/designator
    /// assert_eq!(Particle::LambdaBaryon, Particle::from_str("l"));
    ///
    /// // From the particle number
    /// assert_eq!(Particle::LambdaBaryon, Particle::from_str("10"));
    ///
    /// // From the meshtal tag
    /// assert_eq!(Particle::LambdaBaryon, Particle::from_str("lambda0"));
    ///
    /// // From the full name given in the user manual
    /// assert_eq!(Particle::LambdaBaryon, Particle::from_str("lambda baryon"));
    ///
    /// // Invalid inputs return the Unknown variant
    /// assert_eq!(Particle::Unknown, Particle::from_str("invalid input"));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self::try_from(s).unwrap_or(Self::Unknown)
    }
}

/// Convert from any valid numerical designator
impl TryFrom<u8> for Particle {
    type Error = anyhow::Error;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Unknown),
            1 => Ok(Self::Neutron),
            2 => Ok(Self::Photon),
            3 => Ok(Self::Electron),
            4 => Ok(Self::NegativeMuon),
            5 => Ok(Self::AntiNeutron),
            6 => Ok(Self::ElectronNeutrino),
            7 => Ok(Self::MuonNeutrino),
            8 => Ok(Self::Positron),
            9 => Ok(Self::Proton),
            10 => Ok(Self::LambdaBaryon),
            11 => Ok(Self::PosSigmaBaryon),
            12 => Ok(Self::NegSigmaBaryon),
            13 => Ok(Self::XiBaryon),
            14 => Ok(Self::NegXiBaryon),
            15 => Ok(Self::OmegaBaryon),
            16 => Ok(Self::PosMuon),
            17 => Ok(Self::AntiElectronNeutrino),
            18 => Ok(Self::AntiMuonNeutrino),
            19 => Ok(Self::AntiProton),
            20 => Ok(Self::PosPion),
            21 => Ok(Self::NeuPion),
            22 => Ok(Self::PosKaon),
            23 => Ok(Self::ShortKaon),
            24 => Ok(Self::LongKaon),
            25 => Ok(Self::AntiLambdaBaryon),
            26 => Ok(Self::AntiPosSigmaBaryon),
            27 => Ok(Self::AntiNegSigmaBaryon),
            28 => Ok(Self::AntiNeuXiBaryon),
            29 => Ok(Self::XiBaryon),
            30 => Ok(Self::AntiOmega),
            31 => Ok(Self::Deuteron),
            32 => Ok(Self::Triton),
            33 => Ok(Self::Helion),
            34 => Ok(Self::Alpha),
            35 => Ok(Self::NegPion),
            36 => Ok(Self::NegKaon),
            37 => Ok(Self::HeavyIon),
            _ => Err(anyhow!("Could not infer particle from \"{v}\"")),
        }
    }
}

/// Convert from any valid designator, name, or meshtal output tag
impl TryFrom<&str> for Particle {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let s = s.to_lowercase();

        match s.trim() {
            "0" | "unknown" => Ok(Self::Neutron),
            "1" | "n" | "neutron" => Ok(Self::Neutron),
            "2" | "p" | "photon" => Ok(Self::Photon),
            "3" | "e" | "electron" => Ok(Self::Electron),
            "4" | "|" | "mu_minus" | "negative muon" => Ok(Self::NegativeMuon),
            "5" | "q" | "aneutron" | "anti neutron" => Ok(Self::AntiNeutron),
            "6" | "u" | "nu_e" | "electron neutrino" => Ok(Self::ElectronNeutrino),
            "7" | "v" | "nu_m" | "muon neutrino" => Ok(Self::MuonNeutrino),
            "8" | "f" | "positron" => Ok(Self::Positron),
            "9" | "h" | "proton" => Ok(Self::Proton),
            "10" | "l" | "lambda0" | "lambda baryon" => Ok(Self::LambdaBaryon),
            "11" | "+" | "sigma+" | "positive sigma baryon" => Ok(Self::PosSigmaBaryon),
            "12" | "-" | "sigma-" | "negative sigma baryon" => Ok(Self::NegSigmaBaryon),
            "13" | "x" | "xi0" | "cascade; xi baryon" => Ok(Self::XiBaryon),
            "14" | "y" | "xi_minus" | "negative cascade; negative xi baryon" => {
                Ok(Self::NegXiBaryon)
            }
            "15" | "o" | "omega-" | "omega baryon" => Ok(Self::OmegaBaryon),
            "16" | "!" | "mu_plus" | "positive muon" => Ok(Self::PosMuon),
            "17" | "<" | "anu_e" | "anti electron neutrino" => Ok(Self::AntiElectronNeutrino),
            "18" | ">" | "anu_m" | "anti muon neutrino" => Ok(Self::AntiMuonNeutrino),
            "19" | "g" | "aproton" | "anti proton" => Ok(Self::AntiProton),
            "20" | "/" | "pi_plus" | "positive pion" => Ok(Self::PosPion),
            "21" | "z" | "pi_zero" | "neutral pion" => Ok(Self::NeuPion),
            "22" | "k" | "k_plus" | "positive kaon" => Ok(Self::PosKaon),
            "23" | "%" | "k0_short" | "kaon, short" => Ok(Self::ShortKaon),
            "24" | "^" | "k0_long" | "kaon, long" => Ok(Self::LongKaon),
            "25" | "b" | "alambda0" | "anti lambda baryon" => Ok(Self::AntiLambdaBaryon),
            "26" | "_" | "asigma+" | "anti positive sigma baryon" => Ok(Self::AntiPosSigmaBaryon),
            "27" | "~" | "asigma-" | "anti negative sigma baryon" => Ok(Self::AntiNegSigmaBaryon),
            "28" | "c" | "axi0" | "anti cascade; anti neutral xi baryon" => {
                Ok(Self::AntiNeuXiBaryon)
            }
            "29" | "w" | "xi_plus" | "positive cascade; positive xi baryon" => Ok(Self::XiBaryon),
            "30" | "@" | "aomega-" | "anti omega" => Ok(Self::AntiOmega),
            "31" | "d" | "deuteron" => Ok(Self::Deuteron),
            "32" | "t" | "triton" => Ok(Self::Triton),
            "33" | "s" | "helion" => Ok(Self::Helion),
            "34" | "a" | "alpha" | "alpha particle" => Ok(Self::Alpha),
            "35" | "*" | "pi_minus" | "negative pion" => Ok(Self::NegPion),
            "36" | "?" | "k_minus" | "negative kaon" => Ok(Self::NegKaon),
            "37" | "#" | "heavyion" | "heavy ions" => Ok(Self::HeavyIon),
            _ => Err(anyhow!("Could not infer particle from \"{s}\"")),
        }
    }
}
